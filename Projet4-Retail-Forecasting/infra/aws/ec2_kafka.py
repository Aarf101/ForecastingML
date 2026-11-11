"""Kafka open-source sur EC2 — transport d'événements 100 % dans le compte.

Remplace Kinesis par un **vrai Apache Kafka** (image officielle `apache/kafka`,
mode KRaft, sans Zookeeper) tournant sur une instance EC2 du VPC par défaut.

Pourquoi ce choix : MSK est refusé par le lab (`kafka:ListClustersV2` →
`AccessDeniedException`, aucune politique n'accorde `kafka:*`), or l'énoncé
autorise explicitement « MSK **ou Kafka open-source** ». On reste donc sur Kafka,
auto-hébergé, dans le compte.

Chaîne obtenue :

    producer.py (notebook) ──► Kafka :9092 (EC2) ──► to_s3.py (notebook) ──► S3 raw/sales/

Le producteur et le consommateur sont les **mêmes modules** que ceux validés
contre Redpanda en local : seul `--bootstrap` change.

Usage :
    python -m infra.aws.ec2_kafka up        # SG + instance + Kafka (≈4 min)
    python -m infra.aws.ec2_kafka status    # IP privée + état du conteneur
    python -m infra.aws.ec2_kafka down      # termine l'instance, supprime le SG
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import boto3

from retail_forecast.config import SETTINGS

REGION = SETTINGS.region
TAG = "retail-kafka"
SG_NAME = "retail-kafka-sg"
KEY_NAME = "retail-kafka"  # nom logique, pas de paire de clés (accès via SSM)
AMI_PARAM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"

# Kafka 3.9 (distribution Confluent de Kafka) en mode KRaft mono-nœud.
# Deux pièges rencontrés et contournés ici :
#  * l'image ``apache/kafka`` rejette nos variables d'environnement
#    (« advertised.listeners cannot use the nonroutable meta-address 0.0.0.0 »)
#    alors que la config envoyée est correcte -> on utilise ``confluentinc/cp-kafka``,
#    dont le mapping ``KAFKA_*`` est direct ;
#  * le service de métadonnées d'AL2023 exige IMDSv2, donc ``curl 169.254.169.254``
#    renvoie 401 : l'IP est lue avec ``hostname -I``.
USER_DATA = """#!/bin/bash
set -euxo pipefail
dnf install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user

IP=$(hostname -I | awk '{print $1}')

docker run -d --name kafka --restart always -p 9092:9092 \\
  -e CLUSTER_ID=MkU3OEVBNTcwNTJENDM2Qk \\
  -e KAFKA_NODE_ID=1 \\
  -e KAFKA_PROCESS_ROLES=broker,controller \\
  -e KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093 \\
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://$IP:9092 \\
  -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER \\
  -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT \\
  -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 \\
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \\
  -e KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR=1 \\
  -e KAFKA_TRANSACTION_STATE_LOG_MIN_ISR=1 \\
  -e KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS=0 \\
  -e KAFKA_NUM_PARTITIONS=3 \\
  confluentinc/cp-kafka:7.6.1

# Attente du broker puis création du topic (3 partitions, facteur 1).
for i in $(seq 1 30); do
  if docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then
    docker exec kafka kafka-topics --bootstrap-server localhost:9092 \\
      --create --if-not-exists --topic sales.events \\
      --partitions 3 --replication-factor 1 || true
    break
  fi
  sleep 5
done
"""


def _ec2():
    return boto3.client("ec2", region_name=REGION)


def _default_network(c) -> tuple[str, str]:
    """(subnet_id, vpc_id) : première paire du VPC par défaut."""
    vpcs = c.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        raise SystemExit("aucun VPC par défaut dans ce compte")
    vpc_id = vpcs[0]["VpcId"]
    subnets = c.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "default-for-az", "Values": ["true"]}]
    )["Subnets"]
    subnets.sort(key=lambda s: s["AvailabilityZone"])
    return subnets[0]["SubnetId"], vpc_id


def _ami(c) -> str:
    """AMI Amazon Linux 2023, résolue par paramètre public SSM."""
    ssm = boto3.client("ssm", region_name=REGION)
    return ssm.get_parameter(Name=AMI_PARAM)["Parameter"]["Value"]


def _security_group(c, vpc_id: str) -> str:
    """SG autorisant le port Kafka depuis le VPC uniquement (pas d'Internet)."""
    existing = c.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [SG_NAME]}, {"Name": "vpc-id", "Values": [vpc_id]}]
    )["SecurityGroups"]
    if existing:
        return existing[0]["GroupId"]
    sg = c.create_security_group(
        GroupName=SG_NAME, Description="Kafka open-source (projet retail)", VpcId=vpc_id
    )["GroupId"]
    c.authorize_security_group_ingress(
        GroupId=sg,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 9092,
                "ToPort": 9092,
                "IpRanges": [{"CidrIp": "172.31.0.0/16", "Description": "VPC par defaut"}],
            }
        ],
    )
    return sg


def _instance(c) -> dict | None:
    """Instance taguée ``Name=retail-kafka`` encore vivante, sinon None."""
    res = c.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [TAG]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
        ]
    )["Reservations"]
    return res[0]["Instances"][0] if res else None


def up(instance_type: str, wait: bool = True) -> dict:
    """Crée le SG puis l'instance ; attend que l'état soit ``running``."""
    c = _ec2()
    subnet_id, vpc_id = _default_network(c)
    sg_id = _security_group(c, vpc_id)

    running = _instance(c)
    if running:
        ip = running.get("PrivateIpAddress", "?")
        print(f"[ec2] instance déjà présente : {running['InstanceId']} ({ip})")
        return {"instance_id": running["InstanceId"], "private_ip": ip}

    user_data = USER_DATA  # l'IP n'est connue qu'après création : voir plus bas
    resp = c.run_instances(
        ImageId=_ami(c),
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        SubnetId=subnet_id,
        SecurityGroupIds=[sg_id],
        UserData=user_data.replace("__PRIVATE_IP__", "127.0.0.1"),
        IamInstanceProfile={"Name": "LabInstanceProfile"},
        TagSpecifications=[
            {"ResourceType": "instance", "Tags": [
                {"Key": "Name", "Value": TAG},
                {"Key": "project", "Value": "retail-forecast"},
            ]}
        ],
        BlockDeviceMappings=[
            {"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 16, "VolumeType": "gp3", "DeleteOnTermination": True}}
        ],
    )
    iid = resp["Instances"][0]["InstanceId"]
    print(f"[ec2] instance {iid} demandée ({instance_type})")

    if wait:
        waiter = c.get_waiter("instance_running")
        waiter.wait(InstanceIds=[iid], WaiterConfig={"Delay": 5, "MaxAttempts": 60})
        desc = c.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
        ip = desc.get("PrivateIpAddress", "?")
        print(f"[ec2] running · IP privée {ip}")
        print(f"[ec2] Kafka écoute sur {ip}:9092 (démarrage du conteneur ≈ 60 s)")
        return {"instance_id": iid, "private_ip": ip}
    return {"instance_id": iid, "private_ip": "?"}


def status() -> None:
    """État de l'instance et du conteneur Kafka (via SSM, sans SSH)."""
    c = _ec2()
    inst = _instance(c)
    if not inst:
        print("[ec2] aucune instance retail-kafka")
        return
    iid = inst["InstanceId"]
    state = inst["State"]["Name"]
    ip = inst.get("PrivateIpAddress", "?")
    print(f"[ec2] {iid} {state} · IP privée {ip} · Kafka {ip}:9092")

    ssm = boto3.client("ssm", region_name=REGION)
    try:
        out = ssm.send_command(
            InstanceIds=[iid],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": ["docker ps --format '{{.Names}} {{.Status}}'"]},
        )["Command"]["CommandId"]
        time.sleep(6)
        res = ssm.get_command_invocation(CommandId=out, InstanceId=iid)
        print(f"[ec2] conteneurs : {res.get('StandardOutputContent', '').strip()}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ec2] SSM indisponible ({type(exc).__name__}) — normal si l'instance démarre")


def down() -> None:
    """Termine l'instance puis supprime le groupe de sécurité."""
    c = _ec2()
    inst = _instance(c)
    if inst:
        c.terminate_instances(InstanceIds=[inst["InstanceId"]])
        print(f"[ec2] terminaison demandée pour {inst['InstanceId']}")
    else:
        print("[ec2] aucune instance à terminer")

    try:
        vpc_id = _default_network(c)[1]
        groups = c.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [SG_NAME]}, {"Name": "vpc-id", "Values": [vpc_id]}]
        )["SecurityGroups"]
        for g in groups:
            time.sleep(30)  # l'ENI doit être détachée avant la suppression
            c.delete_security_group(GroupId=g["GroupId"])
            print(f"[ec2] SG {g['GroupId']} supprimé")
    except Exception as exc:  # noqa: BLE001
        print(f"[ec2] SG non supprimé ({str(exc).splitlines()[0][:80]}) — réessayer plus tard")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["up", "status", "down"])
    parser.add_argument("--instance-type", default="t3.medium", help="t3.medium = 4 Go")
    args = parser.parse_args(argv)

    if args.action == "up":
        up(args.instance_type)
        print("\nProducteur / consommateur depuis le notebook (terminal Jupyter) :")
        print("  export KAFKA_BOOTSTRAP=<ip-privee>:9092 S3_BUCKET=" + SETTINGS.bucket)
        print("  python -m retail_forecast.ingest.producer --limit 20000")
        print("  python -m retail_forecast.ingest.to_s3 --max-messages 20000")
    elif args.action == "status":
        status()
    else:
        down()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

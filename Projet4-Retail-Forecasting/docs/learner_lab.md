# Accéder à l'AWS Academy Learner Lab

Le Learner Lab est le compte AWS fourni par l'école : **pas de carte bancaire**, un
crédit de 50 à 100 $, et des sessions de **4 heures** qui se coupent automatiquement.
C'est ce compte qu'il faut utiliser pour les Phases 4 à 6 (SageMaker, Bedrock).

## 1. Recevoir l'accès

Deux cas :

- **Invitation par mail** : un message de *AWS Academy* / *Vocareum* contenant un lien
  d'inscription et un code de cours. Il vient de l'enseignant, pas de vous.
- **Intégration LMS** : le cours apparaît directement dans Moodle / Canvas / Teams de
  l'université (onglet *AWS Academy Learner Lab*).

Si vous n'avez ni l'un ni l'autre : demandez à l'enseignant de vous **enrôler dans le
cours** (il doit ajouter votre adresse dans la section, puis vous recevez le mail).

## 2. Se connecter

1. Ouvrir <https://awsacademy.instructure.com> (ou <https://labs.vocareum.com>).
2. Cliquer **Sign In with AWS Academy** et utiliser l'adresse e-mail **académique**
   (celle qui a reçu l'invitation).
3. Tableau de bord → **Courses** → votre cours → **Modules**.
4. Ouvrir **Learner Lab — Foundational** (parfois nommé *Learner Lab* ou *Sandbox*).

## 3. Démarrer une session

1. Cliquer **Start Lab**. La pastille à côté de **AWS** passe au **vert** (compter
   1 à 3 minutes ; rafraîchir la page si elle reste jaune).
2. Cliquer **AWS** : la console s'ouvre dans un nouvel onglet. C'est le seul moyen
   d'accéder à la console — pas de compte racine, pas de page de facturation.
3. **La session dure 4 heures.** Passé ce délai, tout ce qui tourne (endpoints,
   clusters) est arrêté. Cliquer **End Lab** en fin de séance, sinon le crédit
   continue de tourner.

## 4. Récupérer les identifiants pour le terminal

Dans la page du lab : **AWS Details** → **AWS CLI** → **Show**.

Trois valeurs apparaissent (elles changent **à chaque session**) :

- `aws_access_key_id`
- `aws_secret_access_key`
- `aws_session_token`  ← à ne pas oublier, sinon `InvalidClientTokenId`

### Les installer dans le profil `lab`

Le fichier `~/.aws/config` contient déjà un profil `lab` (région `us-east-1`).
Coller les trois valeurs dans `~/.aws/credentials` :

```ini
[lab]
aws_access_key_id = ASIA...
aws_secret_access_key = ...
aws_session_token = ...
```

Puis vérifier :

```bash
AWS_PROFILE=lab aws sts get-caller-identity
# -> "arn:aws:sts::<compte>:assumed-role/LabRole/<session>"
```

`LabRole` est le rôle pré-créé : c'est lui qu'on met dans `SAGEMAKER_ROLE_ARN`.
Le numéro de compte apparaît dans la réponse ci-dessus.

## 5. Basculer le projet de local vers AWS

Une fois les identifiants collés, un seul script fait le reste :

```bash
make aws-setup            # ou : bash scripts/aws_setup.sh <binome>
```

Il vérifie le profil `lab`, affiche le numéro de compte, crée
`s3://retail-forecast-<compte>-<binome>`, y dépose `data/raw/sales.parquet`
sous `raw/historique/`, et imprime le bloc `.env` à coller.

`.env` est configuré pour le local (s3mock). Pour passer sur AWS, **vider** les
quatre lignes suivantes — sinon boto3 envoie les faux identifiants `test` à AWS :

```ini
S3_ENDPOINT_URL=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_PROFILE=lab
S3_BUCKET=retail-forecast-<compte>-<binome>
SAGEMAKER_ROLE_ARN=arn:aws:iam::<compte>:role/LabRole
```

Retour au local : remettre `S3_ENDPOINT_URL=http://localhost:9090`,
`AWS_ACCESS_KEY_ID=test`, `AWS_SECRET_ACCESS_KEY=test`, et commenter `AWS_PROFILE`.

## 6. Ce qui est limité ou interdit dans le lab

| Limite | Conséquence pour le projet |
|---|---|
| Pas de création d'utilisateurs/roles IAM | utiliser `LabRole`, ne pas toucher à IAM |
| Pas d'accès à la facturation | surveiller le crédit depuis la page du lab |
| **MSK souvent bloqué** | Redpanda en local, ou Kinesis (autorisé) |
| **Bedrock souvent désactivé** | `LLM_PROVIDER=ollama` ou `jumpstart` |
| Pas de QuickSight | Streamlit |
| GPU non disponible par défaut | LightGBM sur CPU (`ml.m5.xlarge`) suffit |
| Certaines instances nécessitent une demande | rester sur `ml.m5.large` / `ml.t3.medium` |
| Session de 4 h | arrêter les endpoints (ils coûtent à l'heure) |

## 7. Réflexes qui protègent le crédit

```bash
# 1. supprimer un endpoint dès qu'on ne s'en sert plus
AWS_PROFILE=lab aws sagemaker list-endpoints --query 'Endpoints[].EndpointName' --output table
AWS_PROFILE=lab aws sagemaker delete-endpoint --endpoint-name <nom>

# 2. lister ce qui tourne encore
AWS_PROFILE=lab aws sagemaker list-training-jobs --status-equals InProgress --output table
```

Coût typique d'une démo complète : **moins de 5 $** (entraînement `ml.m5.large`
≈ 0,115 $/h, endpoint serverless à la requête). Le poste le plus cher serait un
endpoint temps réel laissé allumé toute la nuit.

# Azure deployment plan

The last step of phase 1: run the credit-api container on Azure behind a public HTTPS URL, and have every push to GitHub deploy it. Planned 2026-11-13; started 2026-11-14 in `centralindia`. Created so far: the resource group `rg-credit-api`, the registry `creditapi2611` (`creditapi2611.azurecr.io`, Basic, admin user off), the Container Apps environment `cae-credit-api` and the app `credit-api`, live at https://credit-api.lemonocean-12436fd6.centralindia.azurecontainerapps.io (image `credit-api:manual-1`, pushed by hand). Left: the deploy job. The prediction log stays on the container's disk for now (decided 2026-11-14, see step 4).

**Where things stand:**

- CI tests the API, fetches the champion from Databricks and builds the image ([.github/workflows/credit-api.yml](../.github/workflows/credit-api.yml)).
- The image is thrown away when the runner ends.
- The remaining work: store the image somewhere Azure can pull it from, run it, and add a deploy job.

## Decisions

### Where the image is stored: Azure Container Registry (ACR)

A container registry is to images what Artifactory or Nexus is to jars. CI pushes a tagged image, and Azure pulls it. Every registry below speaks the same protocol (OCI distribution), so `docker push` and `docker pull` work the same; only the name in front of the image changes.

| | ACR (chosen) | GHCR | Docker Hub | Self-hosted `registry` |
|---|---|---|---|---|
| Cost | About $5 a month (Basic) | Free for public images (the repo is public) | Free for public images; private images need a paid plan beyond a small allowance | A server plus storage |
| Azure pulls with | A managed identity: no password anywhere | A GitHub token stored in Azure | A login (anonymous pulls are rate-limited, and cloud services share outgoing addresses) | A login |
| Who can pull | Only what's allowed | Anyone, for a public image | Anyone, for a public image | Only what's allowed |
| Catch | Costs money after the credit | Public image, model included (fine: public Kaggle data) | Nothing gained over GHCR | Running, securing and keeping it up is a project of its own |

**ACR was chosen** for the setup production Azure teams use: a private registry, and Container Apps pulling from it with a managed identity instead of a stored password. GHCR is the fallback if the cost matters. Switching later changes the registry name in the workflow, little else.

### Where the container runs: Azure Container Apps

- **What it gives:** a public HTTPS URL with a certificate; it scales to zero when idle; and it has a monthly free allowance of CPU time and requests, which a demo API should stay under.
- **The cost of scaling to zero:** the first request after an idle spell waits for a cold start, which means pulling the image, starting the container and loading the model. That takes seconds. `--min-replicas 1` avoids it, but then Azure bills for the time the container sits idle.

### Cost and the $200 credit

- **The credit:** a new Azure account's $200 covers any service, but it expires **30 days** after sign-up.
- **After 30 days on the free account:** resources are switched off unless the account is upgraded to pay-as-you-go.
- **After upgrading:** expect about $5 a month (ACR Basic), with Container Apps inside its free allowance.
- **Budget alert first:** Cost Management → Budgets, $10 a month, with emails at 50% and 100%.

## Steps

The names are suggestions. `<region>` is one Azure region for everything (e.g. `centralindia`, `westeurope`, `eastus`). `<acr>` is the registry name: 5 to 50 letters and digits, unique across all of Azure, e.g. `creditapi` plus a few digits.

### 1. Budget alert and the Azure CLI

```
winget install --exact --id Microsoft.AzureCLI
# new terminal, so the PATH applies
az login
az account show --query "{name:name, state:state}" -o table     # the subscription should be Enabled
```

### 2. Resource group and registry

A resource group holds everything this project creates, so it can all be deleted with one command (`az group delete --name rg-credit-api`).

```
az group create --name rg-credit-api --location <region>
# a new subscription has to register each service before its first use (otherwise MissingSubscriptionRegistration);
# the last two are for step 4
az provider register --namespace Microsoft.ContainerRegistry --wait
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az acr create --resource-group rg-credit-api --name <acr> --sku Basic     # the admin user stays off: no registry password
```

Done 2026-11-14: `rg-credit-api` in `centralindia`, registry `creditapi2611`.

### 3. Push the image by hand once

This shows each part before CI automates it. Run it from `credit-api`, after `fetch_champion.py`.

```
az acr login --name <acr>                                  # docker can now push to <acr>.azurecr.io
docker build -t <acr>.azurecr.io/credit-api:manual-1 .
docker push <acr>.azurecr.io/credit-api:manual-1
az acr repository show-tags --name <acr> --repository credit-api -o table
```

Done 2026-11-14: 12 layers pushed, tag `manual-1`.

- **A layer** is one slice of the image's file system: the base image's, plus one per Dockerfile step. A later push uploads only the changed layers.
- **The digest** (`sha256:…`) is a hash of the image's contents, so it names exactly one image forever. A tag is a movable name pointing at a digest, like the `champion` alias pointing at a model version.

**Tag every image with a version, never only `latest`.** Then a tag always means the same image, and rolling back is pointing the app at the previous tag. CI will tag with the commit SHA.

### 4. The container app

```
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights

az containerapp env create --name cae-credit-api --resource-group rg-credit-api --location <region>

az containerapp create --name credit-api --resource-group rg-credit-api --environment cae-credit-api \
  --image <acr>.azurecr.io/credit-api:manual-1 \
  --registry-server <acr>.azurecr.io --registry-identity system \
  --target-port 8000 --ingress external \
  --min-replicas 0 --max-replicas 1

az containerapp show --name credit-api --resource-group rg-credit-api --query properties.configuration.ingress.fqdn -o tsv
```

- **The pull:** `--registry-identity system` gives the app a managed identity and the AcrPull role on the registry. No password is stored.
- **The port:** `--target-port 8000` is the port `fastapi run` listens on. HTTPS on 443 comes from Azure.
- **The check:** `https://<fqdn>/health` should show `workspace.default.credit_risk/1`, and borrower 1 (README, "Run it") should give 0.06459152202678682, as in Docker on the PC.

Done 2026-11-14:

- **The endpoints:** `/health` shows `workspace.default.credit_risk/1`, and borrower 1 gets 0.06459152202678682, the same as on the PC. Both answer in about 0.15 seconds once the app is warm.
- **Plain HTTP** gets a 301 redirect to HTTPS.
- **The first request hung:** `az containerapp create` with `--registry-identity system` first starts a placeholder revision, whose startup probe failed, then replaces it with the real one once the identity can pull. The first request arrived during that swap. `az containerapp revision list` shows which revision is serving.

**A decision at this step: where the prediction log lives.** It's a SQLite file in `/app/data`. On Container Apps, a container's disk is lost on every restart, new version and scale to zero, so the log would be lost with it.

| Option | For | Against |
|---|---|---|
| Accept the loss (demo only) | Nothing to set up | The log is the raw material for monitoring (phase 4) |
| An Azure Files share mounted at `/app/data` | Small change; it costs cents | SQLite on a network share (SMB) can corrupt under concurrent writers. It's only acceptable with `--max-replicas 1` |
| Azure Database for PostgreSQL (Flexible Server, smallest burstable size) | A real database. With SQLModel, the switch is mostly the connection string | Roughly $12–15 a month; one more resource to run |

**Decided 2026-11-14: accept the loss for now.** The log stays on the container's disk, and it's lost on every restart, new version and scale to zero. It comes back when it's needed: for monitoring in phase 4, or for more than one container.

**Several containers:** the API itself scales out with no changes. The model is in the image, and no request depends on another, so raising `--max-replicas` is enough. The log doesn't scale out:
- **SQLite on the container's disk:** each container keeps its own file.
- **SQLite on Azure Files:** several writers over the network share can corrupt the file. Even with one replica, a deploy briefly runs the old and new containers together.

So a shared log needs a database server:
- **PostgreSQL Flexible Server** (recommended): about $12–15 a month, `psycopg` plus a connection string, and pgvector in phase 3.
- **Azure SQL Database's free offer:** free within a monthly allowance, but it needs Microsoft's ODBC driver in the image.

### 5. The deploy job in CI

**Signing in:** GitHub signs in to Azure with **OIDC** (workload identity federation). Azure trusts tokens that GitHub issues for this repo and branch, so no Azure password or secret is stored in GitHub. Set it up once:

- **An identity:** an Entra ID app registration, or a user-assigned managed identity, with a federated credential for `repo:bhaumikchauhan/machine_learning:ref:refs/heads/<branch>`.
- **Its roles:** `AcrPush` on the registry, and permission to update the container app (e.g. `Contributor` on `rg-credit-api`).
- **GitHub secrets:** `AZURE_CLIENT_ID`, `AZURE_TENANT_ID` and `AZURE_SUBSCRIPTION_ID`. These are identifiers, not passwords, but they're kept with the other secrets.

**The workflow:**

- **Permissions:** `permissions: id-token: write` and `contents: read`, so the job can ask GitHub for its OIDC token.
- **The steps:** sign in with the `azure/login` action. Check its current major version first: the `@v4` and `@v5` given for checkout and setup-python were out of date. Then:
  ```
  az acr login --name <acr>
  docker build -t <acr>.azurecr.io/credit-api:${{ github.sha }} .
  docker push <acr>.azurecr.io/credit-api:${{ github.sha }}
  az containerapp update --name credit-api --resource-group rg-credit-api --image <acr>.azurecr.io/credit-api:${{ github.sha }}
  ```
- **Which pushes deploy:** to decide. Deploy only from `main`, and keep the test and build jobs on every branch? The work so far is on the `deployment` branch.
- **After the deploy:** call `/health` and fail the job if the version isn't the one just fetched.

**Phase 1 is done when** a public URL returns predictions from a registered model, and a push to GitHub tests and rebuilds it.

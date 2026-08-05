# Backend deployment (Azure) — no UI

This repo deploys **only the FastAPI backend**. Your UI lives in a separate repo and connects via `VITE_API_BASE_URL` (or equivalent) to the deployed API URL.

## What was added for deployment

| File | Purpose |
|------|---------|
| `Dockerfile` | Backend-only container (excludes `UI/`) |
| `.dockerignore` | Keeps UI, secrets, and dev files out of the image |
| `scripts/startup.sh` | Runs uvicorn on `PORT` (Azure) or `API_PORT` |
| `.github/workflows/deploy-backend-azure.yml` | Build image → ACR → Azure Web App |
| `api/__main__.py` | Reads Azure `PORT` env var |

## Production API URLs

After deploy, your base URL is:

```text
https://<AZURE_WEBAPP_NAME>.azurewebsites.net
```

| Method | Path |
|--------|------|
| `GET` | `/health` |
| `POST` | `/api/v1/jobs` — create job, instant `job_id` (JSON body, no file) |
| `POST` | `/api/v1/jobs/{job_id}/video` — upload video, pipeline runs in background |
| `GET` | `/api/v1/jobs/{job_id}` |
| `WS` | `/api/v1/jobs/{job_id}/stream` |

Docs: `https://<app>.azurewebsites.net/docs`

### Recommended client flow (async)

```text
1. POST /api/v1/jobs          → { job_id, status: "awaiting_upload" }
2. WS  /api/v1/jobs/{id}/stream  → listen for status / step / complete / error
3. POST /api/v1/jobs/{id}/video  → multipart video (do not block UI on this)
```

| Step | Sync (HTTP response) | Async (background + WebSocket) |
|------|----------------------|--------------------------------|
| Create job | Instant `job_id` | — |
      | Upload video | Ack `{ status: "uploading" }` immediately | `uploading` → `uploaded` → pipeline stages |
| Progress | — | `uploading` → `processing` → steps → `complete` |

Legacy single-request upload: `POST /api/v1/jobs/upload` (multipart) — `job_id` only after full upload is received; prefer the two-step flow above.

---

## One-time Azure setup (CLI)

Replace placeholders: `RESOURCE_GROUP`, `LOCATION`, `ACR_NAME`, `APP_PLAN`, `WEBAPP_NAME`.

```bash
az login

az group create --name RESOURCE_GROUP --location LOCATION

az acr create --resource-group RESOURCE_GROUP --name ACR_NAME --sku Basic --admin-enabled true

az appservice plan create \
  --resource-group RESOURCE_GROUP \
  --name APP_PLAN \
  --is-linux \
  --sku S1

az webapp create \
  --resource-group RESOURCE_GROUP \
  --plan APP_PLAN \
  --name WEBAPP_NAME \
  --deployment-container-image-name ACR_NAME.azurecr.io/qsaid-wi-api:latest

az webapp config set \
  --resource-group RESOURCE_GROUP \
  --name WEBAPP_NAME \
  --web-sockets-enabled true

ACR_USER=$(az acr credential show --name ACR_NAME --query username -o tsv)
ACR_PASS=$(az acr credential show --name ACR_NAME --query "passwords[0].value" -o tsv)

az webapp config container set \
  --resource-group RESOURCE_GROUP \
  --name WEBAPP_NAME \
  --docker-custom-image-name ACR_NAME.azurecr.io/qsaid-wi-api:latest \
  --docker-registry-server-url https://ACR_NAME.azurecr.io \
  --docker-registry-server-user "$ACR_USER" \
  --docker-registry-server-password "$ACR_PASS"

az webapp config appsettings set \
  --resource-group RESOURCE_GROUP \
  --name WEBAPP_NAME \
  --settings WEBSITES_PORT=8000 PORT=8000
```

### Application settings (secrets)

Set in Azure Portal → Web App → **Configuration**, or via CLI:

```bash
az webapp config appsettings set \
  --resource-group RESOURCE_GROUP \
  --name WEBAPP_NAME \
  --settings \
    AZURE_OPENAI_ENDPOINT="https://YOUR-RESOURCE.openai.azure.com/" \
    AZURE_OPENAI_KEY="..." \
    AZURE_OPENAI_API_VERSION="2025-01-01-preview" \
    AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o" \
    AZURE_OPENAI_WHISPER_ENDPOINT="https://..." \
    AZURE_OPENAI_WHISPER_KEY="..." \
    AZURE_OPENAI_WHISPER_DEPLOYMENT="whisper" \
    AZURE_STORAGE_ACCOUNT_NAME="..." \
    AZURE_STORAGE_ACCOUNT_KEY="..." \
    AZURE_STORAGE_CONTAINER_NAME="document-videos" \
    AZURE_STORAGE_JOB_PREFIX="jobs" \
    AZURE_BLOB_SAS_EXPIRY_HOURS="24" \
    CORS_ORIGINS="https://your-ui-domain.com" \
    API_HOST="0.0.0.0" \
    PORT="8000"
```

**CORS:** Set `CORS_ORIGINS` to your UI origin(s), comma-separated.

**Upload size:** Increase max request body size (e.g. 500 MB) for large videos.

**Always On:** Enable so background jobs are not suspended on idle.

---

## GitHub Actions deploy

### Service principal

```bash
az ad sp create-for-rbac \
  --name "qsaid-wi-github-deploy" \
  --role contributor \
  --scopes /subscriptions/SUBSCRIPTION_ID/resourceGroups/RESOURCE_GROUP \
  --sdk-auth
```

### GitHub secrets

| Secret | Value |
|--------|--------|
| `AZURE_CREDENTIALS` | JSON from service principal command |
| `AZURE_WEBAPP_NAME` | Web app name |
| `ACR_LOGIN_SERVER` | `ACR_NAME.azurecr.io` |
| `ACR_USERNAME` | ACR admin username |
| `ACR_PASSWORD` | ACR admin password |

Push to `main` or run the workflow manually.

---

## Manual Docker deploy

```bash
az acr login --name ACR_NAME
docker build -t ACR_NAME.azurecr.io/qsaid-wi-api:latest .
docker push ACR_NAME.azurecr.io/qsaid-wi-api:latest
```

---

## Local Docker test

```bash
docker build -t qsaid-wi-api .
docker run --rm -p 8000:8000 --env-file .env -e PORT=8000 qsaid-wi-api
curl http://localhost:8000/health
```

---

## Separate UI repo

```env
VITE_API_BASE_URL=https://WEBAPP_NAME.azurewebsites.net
```

---

## Verify

```bash
curl https://WEBAPP_NAME.azurewebsites.net/health
```

Expected: `{"status":"ok","service":"qsaid-wi","version":"1.0.0"}`

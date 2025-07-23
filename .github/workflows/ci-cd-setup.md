# [CI/CD] Configure GitHub Actions Azure Authentication with OIDC + Federated Identity Credentials

This document describes how to configure **OIDC (OpenID Connect)** based authentication in GitHub Actions to securely access Azure resources, without storing secrets.

---

## Step 1: Login to Azure and Check Subscription ID

```bash
az login --use-device-code
az account show --query id -o tsv
```

---

## Step 2: Register Azure App and Set Up Federated Credentials

Run the following script in Ubuntu to create the Azure App, add federated credentials, and assign permissions:

```bash
#!/bin/bash

# Basic variables
APP_NAME="GitHub-OIDC-App"
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)
RESOURCE_GROUP="plan-search-rg"
ACR_NAME="<YOUR_ACR_NAME>"
ROLE_NAME="Contributor"

# 1. Create app registration
APP_ID=$(az ad app create \
  --display-name "$APP_NAME" \
  --query appId -o tsv)

# 2-1. Add federated credential for main branch
az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "github-oidc-branch",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:Azure/plan-search-chatbot:ref:refs/heads/main",
  "description": "GitHub Actions OIDC federation for main branch",
  "audiences": ["api://AzureADTokenExchange"]
}'

# 2-2. Add federated credential for production environment
az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "github-oidc-env",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:Azure/plan-search-chatbot:environment:production",
  "description": "GitHub Actions OIDC federation for production environment",
  "audiences": ["api://AzureADTokenExchange"]
}'

# 3. Create service principal
SP_OBJECT_ID=$(az ad sp create --id "$APP_ID" --query id -o tsv)

# 4. Assign Contributor role at resource group scope
az role assignment create --assignee-object-id "$SP_OBJECT_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "$ROLE_NAME" \
  --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP"

# 5. Assign ACR permissions
ACR_ID=$(az acr show -n "$ACR_NAME" -g "$RESOURCE_GROUP" --query id -o tsv)
az role assignment create --assignee-object-id "$SP_OBJECT_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "User Access Administrator" \
  --scope "$ACR_ID"

echo "✅ Setup complete: App ID = $APP_ID, Object ID = $SP_OBJECT_ID"
```

---

> After setup, GitHub Actions can authenticate to Azure using OIDC securely, without storing any secret keys.

| Secret Name                                | Description                                               |
|--------------------------------------------|-----------------------------------------------------------|
| AZURE_OPENAI_API_KEY                       | from `backend-env.json`                                   |
| AZURE_OPENAI_ENDPOINT                      | from `backend-env.json`                                   |
| AZURE_OPENAI_DEPLOYMENT_NAME               | from `backend-env.json`                                   |
| AZURE_OPENAI_QUERY_DEPLOYMENT_NAME         | from `backend-env.json`                                   |
| BING_GROUNDING_PROJECT_ENDPOINT            | from `backend-env.json`                                   |
| BING_GROUNDING_CONNECTION_ID               | from `backend-env.json`                                   |
| BING_GROUNDING_AGENT_MODEL_DEPLOYMENT_NAME | from `backend-env.json`                                   |
| BING_API_KEY                               | Bing API key                                              |
| SLACK_WEBHOOK_URL                          | Slack webhook URL                                         |
| ACR_NAME                                   | Azure Container Registry name                             |
| AZURE_CLIENT_ID                            | Application (client) ID from App Registration             |
| AZURE_SUBSCRIPTION_ID                      | Azure Subscription ID                                     |
| AZURE_TENANT_ID                            | Azure Tenant ID                                           |
| BACKEND_ENV_JSON                           | Content of `backend-env.json`                             |
| MAIN_PARAMETERS_JSON                       | Content of `main.parameters.json`                         |

---

## Step 4: Create GitHub Environment

To use the production environment in GitHub Actions, create a new environment named `production`.
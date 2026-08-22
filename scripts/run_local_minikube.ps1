# Windows / PowerShell equivalent of run_local_minikube.sh (demo recording helper).
param([string]$ImageTag = "local")

$ErrorActionPreference = "Stop"
$Image = "cats-vs-dogs-mlops:$ImageTag"

if (-not (Get-Command minikube -ErrorAction SilentlyContinue)) {
    throw "minikube is not installed. Install Docker Desktop + minikube first."
}

minikube status *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[local] starting minikube ..."
    minikube start --cpus=2 --memory=4096
}

Write-Host "[local] pointing docker at minikube ..."
& minikube -p minikube docker-env --shell powershell | Invoke-Expression

Write-Host "[local] building $Image ..."
docker build -t $Image .

Write-Host "[local] deploying ..."
kubectl apply -f k8s/base/namespace.yaml
kubectl apply -f k8s/base/deployment.yaml
kubectl apply -f k8s/base/service.yaml
kubectl apply -f k8s/monitoring/prometheus.yaml
kubectl apply -f k8s/monitoring/grafana.yaml
kubectl -n mlops set image deployment/cats-vs-dogs-api api=$Image
kubectl -n mlops rollout status deployment/cats-vs-dogs-api --timeout=300s

$Url = (minikube service cats-vs-dogs-api -n mlops --url) | Select-Object -First 1
Write-Host "[local] service URL: $Url"

bash ./scripts/smoke_test.sh $Url
python -m src.monitoring.performance_check --url $Url --sample-size 20

Write-Host ""
Write-Host "  API        : $Url/docs"
Write-Host "  Metrics    : $Url/metrics"
Write-Host "  Prometheus : http://$(minikube ip):30090"
Write-Host "  Grafana    : http://$(minikube ip):30300"

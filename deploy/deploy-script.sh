#!/bin/bash

# Music Lyrics Fine-tuning Deployment Script for OpenShift AI
# Project: lyric-professor
# Model: EleutherAI/pythia-410m for lyrics understanding

set -e  # Exit on any error

# Configuration
REGISTRY_URL=${REGISTRY_URL:-"ghcr.io"}
IMAGE_NAME=${IMAGE_NAME:-"thesteve0/pythia-finetune"}

echo "🚀 Deploying Pythia-410M lyrics fine-tuning job to OpenShift AI..."

# Step 1: Generate timestamp-based version
echo "📅 Generating timestamp-based version..."
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
NEW_VERSION="0.1.${TIMESTAMP}"
echo "New version: ${NEW_VERSION}"

# Update .version file
echo "${NEW_VERSION}" > ../.version
echo "✅ Updated .version file"

# Full image name with new version
FULL_IMAGE_NAME="${REGISTRY_URL}/${IMAGE_NAME}:${NEW_VERSION}"
echo "Container image: ${FULL_IMAGE_NAME}"

# Step 2: Build and push container image
echo "📦 Building container image (forcing fresh build)..."
cd .. # Go to project root
docker build -t "${FULL_IMAGE_NAME}" .

echo "📤 Pushing container image to registry..."
docker push "${FULL_IMAGE_NAME}"
echo "✅ Container image built and pushed successfully"

# Step 3: Update PyTorchJob with new version and image
echo "🔧 Updating PyTorchJob configuration..."
cd deploy

# Create a temporary copy of the PyTorchJob to modify
cp pytorchjob.yaml pytorchjob-temp.yaml

# Add project-version label to metadata
sed -i "/ml-platform\/workbench:/a\\    project-version: \"${NEW_VERSION}\"" pytorchjob-temp.yaml

# Update the container image
sed -i "s|image: your-registry/pythia-finetune:.*|image: ${FULL_IMAGE_NAME}|g" pytorchjob-temp.yaml

echo "✅ PyTorchJob updated with version ${NEW_VERSION}"

# Step 4: Ensure we're in the correct namespace IT IS ASSUMED YOU ALREADY DID THIS
#echo "🎯 Setting up OpenShift AI namespace..."
#kubectl config set-context --current --namespace=lyric-professor

# Check for existing PyTorchJob and clean up if necessary
echo "🔍 Checking for existing PyTorchJob deployments..."
if kubectl get pytorchjob pythia-finetuning-demo >/dev/null 2>&1; then
    echo "⚠️  Found existing PyTorchJob 'pythia-finetuning-demo', deleting..."
    kubectl delete pytorchjob pythia-finetuning-demo
    
    # Wait for pods to be cleaned up
    echo "Waiting for pods to be cleaned up..."
    sleep 10
    
    # Check if any pods are still running and force delete if necessary
    if kubectl get pods -l pytorch-job-name=pythia-finetuning-demo --no-headers 2>/dev/null | grep -v "No resources found"; then
        echo "Force deleting remaining pods..."
        kubectl delete pods -l pytorch-job-name=pythia-finetuning-demo --force --grace-period=0 2>/dev/null || true
        sleep 5
    fi
    
    echo "✅ Cleanup completed"
else
    echo "✅ No existing PyTorchJob found"
fi

# Create storage PVCs first (required before PyTorchJob)
echo "Creating storage PVCs for training data, models, and workspace..."
kubectl apply -f storage.yaml

echo "PVCs created will bind when training pod starts"
kubectl get pvc

# Apply the PyTorchJob to start the training
echo "Starting PyTorchJob for lyrics fine-tuning with version ${NEW_VERSION}..."
kubectl apply -f pytorchjob-temp.yaml

# Wait a moment for the job to initialize
sleep 5

# Check the status of the PyTorchJob
echo "Checking PyTorchJob status..."
kubectl get pytorchjob pythia-finetuning-demo -o wide

# Show the pods created by the job
echo "Listing training pods..."
kubectl get pods -l pytorch-job-name=pythia-finetuning-demo

# Display resource usage information
echo "Resource allocation for single-node training:"
echo "- Target VRAM usage: 45GB/48GB (94% utilization)"
echo "- RAM: 32GB"
echo "- CPU: 8 vCPU"
echo "- GPU: 1x NVIDIA L40S (48GB VRAM)"

echo "\nTo monitor training progress:"
echo "kubectl logs -f pythia-finetuning-demo-master-0"
echo "\nTo check OpenShift AI metrics:"
echo "Navigate to: OpenShift AI → Distributed workloads → Project metrics"

# Follow the logs of the training pod
echo "\nFollowing training logs..."
kubectl logs -f pythia-finetuning-demo-master-0
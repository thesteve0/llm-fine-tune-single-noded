# Apply the YAML to start the job
kubectl apply -f pytorchjob.yaml

# Check the status of the PyTorchJob
kubectl get pytorchjob pythia-finetuning-demo

# Check the logs of the training pod to see the progress
kubectl logs -f pythia-finetuning-demo-master-0
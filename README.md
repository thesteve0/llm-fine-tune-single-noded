# llm-fine-tune-single-noded

Music Lyrics Fine-tuning with OpenShift AI - Single Node Training

This project demonstrates fine-tuning EleutherAI/pythia-410m on music lyrics using Red Hat OpenShift AI with PyTorchJob orchestration.

## Quick Start

1. **Deploy Training Job**:
   ```bash
   cd deploy
   ./deploy-script.sh
   ```

2. **Monitor Training**:
   ```bash
   kubectl logs -f pythia-finetuning-demo-master-0
   ```

3. **Download Trained Model** (after training completes):
   
   Create a temporary pod to access the PVC:
   ```bash
   cat << EOF > temp-pod.yaml
   apiVersion: v1
   kind: Pod
   metadata:
     name: model-extractor
   spec:
     containers:
     - name: extractor
       image: registry.redhat.io/ubi8/ubi:latest
       command: ["sleep", "3600"]
       volumeMounts:
       - name: models
         mountPath: /models
     volumes:
     - name: models
       persistentVolumeClaim:
         claimName: trained-models-pvc
     restartPolicy: Never
   EOF
   
   oc apply -f temp-pod.yaml
   ```
   
   Download the model files:
   ```bash
   # Wait for pod to be ready
   oc wait --for=condition=Ready pod/model-extractor
   
   # Sync the trained model to local directory
   oc rsync model-extractor:/models/ ./downloaded-model/
   
   # Optional: Browse available files first
   oc exec model-extractor -- ls -la /models
   
   # Clean up
   oc delete pod model-extractor
   rm temp-pod.yaml
   ```

## What You'll Get

After training completes, your downloaded model directory will contain:

- `best_model/` - Your fine-tuned Pythia-410m model
- `training_summary.json` - Training configuration and results  
- `training_history.json` - Epoch-by-epoch metrics
- `demo_outputs/sample_responses.json` - Sample model outputs
- `checkpoints/` - Training checkpoints

## Project Details

See [claude.md](claude.md) for complete project documentation, technical specifications, and OpenShift AI integration details.
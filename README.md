# Evaluation of lora checkpoints

This repo is used to evaluate lora checkpoints.

Steps:

1. Ensure you have changed to the right params at the top part of lora_testing.py

2. Ensure you have mounted the correct volumns using the docker-compose.yml (namely pretrained_models, data and checkpoints folder)

3. Build the container with docker compose:

```bash
docker compose build whisper_lora_testing
```

4. Run the container with docker compose:

```bash
docker compose run whisper_lora_testing
```

5. Run lora_testing.py in the container:

```bash
python3 lora_testing.py
```

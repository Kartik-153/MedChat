# MedChat

## Steps to run the project

1. Clone the repository
```bash
git clone https://github.com/Kartik-153/MedChat.git
```
2. Create a virtual enviroment

```bash
conda create -n <env name> python -y
```

```bash
conda activate <env name>
```

3. Install requirements
```bash
pip install -r requirements.txt
```

4. Create Pinecone API key and save the api key and index name into .env file in the root directory
```ini
PINECONE_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxx"
PINECOME_INDEX_NAME="xxxxxxxxxxxxxxxxxxxxxxxx"
```

## Dowmload the model from instruction file in data directory
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

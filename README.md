---
title: GapFinder
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# GapFinder

O **GapFinder** é um MVP (Minimum Viable Product) de Inteligência Artificial desenvolvido por **Vilker Zucolotto Pessin** como parte de sua pesquisa de doutorado, com o objetivo de auxiliar pesquisadores na identificação de lacunas científicas em artigos acadêmicos.

**Demo online:** https://huggingface.co/spaces/dmgobbi/GapFinder

**Artigo científico:** https://www.scholink.org/ojs/index.php/selt/article/view/55751

## Stack Tecnológico

- **Frontend**: Streamlit
- **Orquestração**: LangChain
- **Vector Store**: ChromaDB (ephemeral)
- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2 (50+ idiomas)
- **LLM Local**: Qwen 2.5-3B-Instruct (GGUF q4_k_m) via llama-cpp-python (requer GPU)
- **LLM Cloud**: Gemini 2.5 Flash Lite (BYOK — traga sua própria chave)
- **Extração de PDF**: PyMuPDF
- **Exportação**: pandas + XlsxWriter (CSV/Excel)
- **GPU**: NVIDIA CUDA 12.1 (T4 ou superior)

## Funcionalidades

- Upload e processamento de múltiplos PDFs simultaneamente
- Extração automática de metadados (DOI, autor, título)
- Pipeline RAG: chunking → retrieval semântico → análise por LLM
- Dois modos de análise: **Local** (Qwen 2.5-3B, requer GPU) ou **Cloud** (Gemini API, BYOK)
- Detecção automática de GPU — modo Local habilitado apenas com CUDA disponível
- Classificação estruturada de lacunas: tipo, descrição, evidência, sugestão
- Exportação de resultados para CSV e Excel

## Instalação

### Setup Local

```bash
# Clone o repositório
git clone https://github.com/vilkerpessin/GapFinder.git
cd GapFinder

# Crie e ative um ambiente virtual
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# (Modo local) Baixe o modelo GGUF:
mkdir -p models
# Baixe qwen2.5-3b-instruct-q4_k_m.gguf de Qwen/Qwen2.5-3B-Instruct-GGUF no Hugging Face
# e coloque em models/
```

### Executar Localmente

```bash
streamlit run app.py --server.port=7860
```

Acesse: `http://localhost:7860`

Para rodar os testes: `pytest tests/ -v`

## Uso

1. Faça upload de um ou mais arquivos PDF de artigos científicos
2. Escolha o modo de análise na barra lateral: **Local LLM** (requer GPU) ou **Cloud (Gemini)**
3. Se Cloud, insira sua chave da API Gemini (obtenha gratuitamente em [Google AI Studio](https://aistudio.google.com/app/apikey))
4. Clique em "Analyze Papers" — o sistema ingere o PDF, recupera contexto relevante e gera insights via LLM
5. Cada lacuna identificada inclui: tipo, descrição, citação do texto e sugestão de pesquisa
6. Exporte os resultados para CSV ou Excel

## Autor

- Nome: Vilker Zucolotto Pessin
- E-mail: vilker.pessin@gmail.com


## Contribuindo

Este projeto é open source e está aberto a contribuições. Veja o arquivo [CONTRIBUTING.md](CONTRIBUTING.md) para instruções detalhadas.


## Licença

Este projeto é distribuído sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.


## Agradecimentos

Agradeço imensamente a todos os pesquisadores(as), interessados(as) e desenvolvedores(as) que acreditam no potencial do GapFinder como ferramenta para fortalecer o avanço da pesquisa científica.
Sua participação é essencial para que possamos construir uma ciência mais aberta, colaborativa e ética.

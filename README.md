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
- **LLM Modal**: Qwen 2.5-7B-Instruct via Modal.com (A10G GPU, sem necessidade de chave)
- **LLM Cloud**: Gemini 2.5 Flash Lite (BYOK — traga sua própria chave)
- **Extração de PDF**: PyMuPDF
- **Exportação**: pandas + XlsxWriter (CSV/Excel)

## Funcionalidades

- Upload e processamento de múltiplos PDFs simultaneamente
- Extração automática de metadados (DOI, autor, título)
- Pipeline RAG: chunking → retrieval semântico → análise por LLM
- Dois modos de análise: **Modal** (Qwen 2.5-7B, sem GPU local) ou **Cloud** (Gemini API, BYOK)
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
```

### Executar Localmente

```bash
# Modo Modal (requer endpoint Modal deployado)
export MODAL_INFERENCE_URL="https://your-endpoint.modal.run"
streamlit run app.py --server.port=7860

# Modo Cloud (Gemini) não requer variável de ambiente
streamlit run app.py --server.port=7860
```

Acesse: `http://localhost:7860`

Para rodar os testes: `pytest tests/ -v`

### Deploy do servidor Modal (inferência Qwen)

```bash
# Requer venv separado com modal instalado
source ~/.venvs/modal-tools/bin/activate
modal deploy modal_inference.py
# Copie a URL gerada para MODAL_INFERENCE_URL
```

## Uso

1. Faça upload de um ou mais arquivos PDF de artigos científicos
2. Escolha o modo de análise na barra lateral: **Cloud (Gemini)** (recomendado, melhor qualidade) ou **Modal (Qwen 2.5-7B)** (sem chave necessária)
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

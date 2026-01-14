# GapFinder

O **GapFinder** é um MVP (Minimum Viable Product) de Inteligência Artificial desenvolvido por **Vilker Zucolotto Pessin** como parte de sua pesquisa de doutorado, com o objetivo de auxiliar pesquisadores na identificação de lacunas científicas em artigos acadêmicos.

**Artigo científico:** https://www.scholink.org/ojs/index.php/selt/article/view/55751

## Funcionalidades

- Upload e processamento de múltiplos PDFs simultaneamente
- Extração automática de metadados (DOI, autor, título)
- Detecção de palavras-chave relacionadas a lacunas de pesquisa
- Análise de sentimento (VADER) para avaliar criticidade
- Exportação de resultados para Excel

## Instalação

### Requisitos

- Python 3.11+
- pip

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

# Baixe os dados do NLTK necessários
python -c "import nltk; nltk.download('vader_lexicon')"

# Configure a SECRET_KEY (obrigatório para produção)
# Gere uma chave segura:
python -c "import secrets; print(secrets.token_hex(32))"

# Copie o arquivo de exemplo e configure:
cp .env.example .env
# Edite .env e substitua SECRET_KEY pelo valor gerado acima
```

### Executar Localmente

```bash
python app.py
```

Acesse: `http://localhost:5000`

## Uso

1. Faça upload de um ou mais arquivos PDF de artigos científicos
2. O sistema identifica parágrafos contendo palavras-chave relacionadas a lacunas de pesquisa
3. Cada parágrafo recebe um "Insight Score" calculado via análise de sentimento
4. Os resultados podem ser visualizados na interface ou exportados para Excel

## Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar com cobertura
pytest tests/ --cov=app --cov-report=html
```

## Variáveis de Ambiente

- `SECRET_KEY`: Chave secreta para sessões Flask (obrigatório em produção)
- `FLASK_ENV`: Ambiente de execução (`development` ou `production`)


## Autor

- Nome: Vilker Zucolotto Pessin
- E-mail: vilker.pessin@gmail.com


## Contribuindo

Este projeto é open source e está aberto a contribuições.
Se você deseja colaborar:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nome-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona funcionalidade X'`)
4. Push para a branch (`git push origin feature/nome-feature`)
5. Abra um Pull Request


## Licença

Este projeto é distribuído sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.


## Agradecimentos

Agradeço imensamente a todos os pesquisadores(as), interessados(as) e desenvolvedores(as) que acreditam no potencial do GapFinder como ferramenta para fortalecer o avanço da pesquisa científica.
Sua participação é essencial para que possamos construir uma ciência mais aberta, colaborativa e ética.
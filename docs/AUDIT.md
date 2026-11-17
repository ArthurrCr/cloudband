# Auditorias de dados e de avaliação

Verificações feitas sobre os dados e sobre a avaliação: o que motivou cada uma, as hipóteses testadas, a
evidência, a causa encontrada e como reproduzir. Decisões de protocolo ficam em `protocols/DIVERGENCES.md`;
limitações que restam, em `LIMITATIONS.md`.

---

## A1. Recorte do patch de 509 pixels do CloudSEN12+ (15/11/2026)

**Resultado:** defeito encontrado, corrigido e a correção confirmada nos dados reais (17/11/2026). O recorte de
512 para 509 pixels tirava o lado errado em um dos eixos. Registrado como DIVERGENCES D17; a limitação que resta
está em LIMITATIONS L1.

### Sintoma

Os gráficos de BOA por faixa de pixels da classe (teste L1C, convenção do CloudSEN12) mostraram os modelos
treinados muito abaixo das quatro versões publicadas do OmniCloudMask **só** na faixa clear 0–10%. Nessa faixa
havia n = 162 patches, contra n = 57 num gráfico equivalente feito antes, no projeto anterior
(Cloud-Segmentation); nas outras faixas os dois números ficavam próximos.

### O que foi verificado

Todos os números abaixo vêm das contagens de confusão salvas por patch no teste L1C (975 patches), calculadas
com o recorte antigo: `phase2-v2-counts-test-ocm-rgn-published-v1.7.0-model{1.0,4.0}.csv` e
`phase2-v2-counts-test-seed0.csv`, em `cloudband_treino/results`.

**1. Os modelos são avaliados sobre os mesmos patches e o mesmo rótulo.** Os seis modelos (as versões publicadas
1.0 e 4.0, regnety, convnext, o ensemble OCM e o Swin) cobrem os mesmos 975 patches, e `class_share` confirma que
todos têm a mesma contagem de pixels de cada classe em cada patch (ela recusa as contagens se algum modelo
divergir). A diferença não vinha de patches ou rótulos diferentes entre os modelos.

**2. De onde vem a diferença na faixa clear 0–10%.** Mediana sobre os 162 patches:

| Modelo | BOA | Sensibilidade (pixels de céu limpo achados) | Especificidade |
|---|---|---|---|
| OCM publicado 1.0 | 0,952 | 0,957 | 0,995 |
| OCM publicado 4.0 | 0,954 | 0,967 | 0,995 |
| regnety treinado | 0,916 | 0,861 | 0,993 |
| Swin treinado | 0,879 | 0,793 | 0,996 |
| OCM treinado (ensemble) | 0,868 | 0,776 | 0,992 |
| convnext treinado | 0,835 | 0,682 | 0,991 |

A diferença está quase toda na sensibilidade, ou seja, em achar os poucos pixels de céu limpo. A mediana de céu
limpo nesses patches é de 1,52%.

**3. A assinatura do preenchimento.** A contagem de pixels de céu limpo por patch (`clear_pixel_audit`) mostra:

| Medida | Valor |
|---|---|
| Pixels anotados por patch | 259.081 em todos (509 × 509) |
| Menor número de pixels de céu limpo num patch | **1.527** |
| Patches sem nenhum pixel de céu limpo | **0** |
| Patches com exatamente 1.527 pixels de céu limpo | **67** |
| Patches só de céu limpo | 199 |

1.527 = 3 × 509: uma faixa de 3 colunas (ou linhas) atravessando o patch. Nenhum patch tem menos que isso, e 67
têm exatamente isso; num conjunto de 975 patches com cenas totalmente nubladas, isso só se explica por uma faixa
fixa rotulada como céu limpo presente em todo patch.

**4. A faixa é preenchimento: imagem zerada.** Nos 67 patches cujo único "céu limpo" é a faixa:

| Modelo | Pixels da faixa marcados como céu limpo (de 1.527) |
|---|---|
| OCM publicado 4.0 | 1.527 em todos os 67 |
| regnety treinado | média 1.387 (mínimo 1.111) |
| Swin treinado | média 1.248 (mínimo 518) |
| OCM treinado (ensemble) | média 1.214 (mínimo 998) |
| convnext treinado | média 1.109 (mínimo 671) |

O OmniCloudMask 1.7.0 trata como "sem dado" todo pixel em que todas as bandas valem `no_data_value` (0 por
padrão) e zera a previsão nele, ou seja, devolve a classe 0, céu limpo: `predict_from_array` tem
`no_data_value=0` e `apply_no_data_mask=True` por padrão, e `raster_utils.mask_prediction` faz
`mask = ~np.all(scene == no_data_value, axis=0)` e `pred_tracker_np *= mask` [4]. O nosso `ocm.predict_array` não
altera esses padrões. Acertar 100% da faixa em todos os patches só acontece se a imagem ali for toda zero, isto
é, preenchimento.

**5. O lado do preenchimento.** O dataset card do CloudSEN12+ diz: *"The padding is filled with zeros in the left
and bottom sides of the image"* [3], e o código seguia isso (linhas 0–508 e colunas 3–511). As figuras salvas no
notebook `baseline.ipynb` do projeto anterior [5], que plotam o canvas inteiro de 512, mostram a borda preta **à
direita e embaixo** (cópia em `evidence/cloudsen12_canvas_512_padding.png`). A segunda figura mostra também a
borda do rótulo como céu limpo. O recorte tirava as linhas certas (as 3 de baixo), mas as colunas erradas: perdia
3 colunas reais à esquerda e mantinha as 3 colunas de preenchimento da direita. Os artigos do CloudSEN12 [1] e do
CloudSEN12+ [2] não descrevem o preenchimento.

### Causa

`datasets/cloudsen12.py` usava `VALID_COLUMNS = slice(3, 512)`, seguindo o dataset card. Como o leitor GDAL, o
leitor rápido e o cache usam a mesma `valid_window`, o defeito atingia treino, validação e teste.

### Correção

- `VALID_COLUMNS = slice(0, 509)`, com o motivo comentado no código.
- `check_padding` recusa, em toda leitura, uma imagem cuja parte descartada não seja toda zero. Um recorte do lado
  errado agora falha na leitura do primeiro patch.
- `CROP_REVISION = 2`: as contagens salvas (`phase2-v2-crop2-...`) e o cache local (`.../crop2`) levam a revisão
  no nome, então nada calculado com o recorte antigo é relido.
- Saíram do notebook as conferências contra resultados calculados com o recorte antigo.
- Testes: o recorte mantém as primeiras colunas e descarta as últimas; `check_padding` recusa o layout do dataset
  card; os patches sintéticos dos testes do leitor rápido têm o preenchimento à direita e embaixo.

### Como reproduzir

- **Antes da correção:** `evaluation.clear_pixel_audit(evaluation.load_counts(<um dos arquivos sem crop2>))` dá a
  tabela do item 3.
- **Depois da correção:** a célula "Verificação do recorte" do notebook `00_baselines/score_ocm_cloudsen12.ipynb`
  (`phase2_eval.crop_check(published)`) roda a mesma auditoria nas contagens novas, em L1C e L2A. Com o recorte
  certo, a menor contagem de céu limpo deve ser 0 (os 67 patches totalmente nublados) e nenhuma contagem deve se
  repetir em dezenas de patches.

### Confirmação depois da correção (17/11/2026)

O notebook de avaliação foi rodado de novo com o recorte corrigido (CROP_REVISION 2). As contagens novas estão em
`cloudband_treino/results`, com `crop2` no nome: `phase2-v2-crop2-counts-test-ocm-rgn-published-v1.7.0-model{1.0,2.0,3.0,4.0}.csv`
e `phase2-v2-crop2-counts-test-seed{0..4}.csv` (e as equivalentes `test_l2a`). Comparação no teste L1C, com as
mesmas contagens do item 3 de um lado e as contagens novas do outro.

**A faixa de preenchimento sumiu** (`clear_pixel_audit` sobre o OCM publicado 4.0):

| Medida | Recorte antigo | Recorte corrigido |
|---|---|---|
| Pixels anotados por patch | 259.081 em todos | 259.081 em todos |
| Menor número de pixels de céu limpo num patch | 1.527 | **0** |
| Patches sem nenhum pixel de céu limpo | 0 | **67** |
| Patches com exatamente 1.527 (3 × 509) pixels de céu limpo | 67 | **0** |
| Patches só de céu limpo | 199 | 199 |

Os 67 patches que antes tinham exatamente a faixa são agora os 67 totalmente nublados, e os 199 patches só de céu
limpo não mudam: a faixa era a única diferença.

**A discrepância na faixa clear 0–10% desapareceu** (mediana sobre os patches com BOA definido; modelos treinados na
seed 0, para comparar com a mesma base do item 2):

| Modelo | n antes | BOA antes | Sensibilidade antes | n depois | BOA depois | Sensibilidade depois |
|---|---|---|---|---|---|---|
| OCM publicado 1.0 | 162 | 0,952 | 0,957 | 97 | 0,841 | 0,699 |
| OCM publicado 4.0 | 162 | 0,954 | 0,967 | 97 | 0,872 | 0,771 |
| OCM treinado (ensemble) | 162 | 0,868 | 0,776 | 97 | 0,870 | 0,762 |
| regnety treinado | 162 | 0,916 | 0,861 | 97 | 0,844 | 0,708 |
| convnext treinado | 162 | 0,835 | 0,682 | 97 | 0,849 | 0,781 |
| Swin treinado | 162 | 0,879 | 0,793 | 97 | 0,828 | 0,695 |

As versões publicadas caíram de cerca de 0,95 para 0,84–0,87: os pontos que ganhavam ao marcar a faixa zerada como
céu limpo, pela máscara de no-data do pacote [4], deixaram de existir. Os modelos treinados mudaram pouco, porque
acertavam só parte da faixa. Com o recorte corrigido, o OCM treinado (0,870) fica praticamente empatado com o OCM
publicado 4.0 (0,872) nessa faixa, e acima da versão 1.0. A diferença de 8 a 12 pontos vista antes era um artefato
do recorte, não uma diferença entre os modelos.

### Efeito na faixa clear 0–10% e o n = 57

Sem a faixa, os 67 patches totalmente nublados ficam sem céu limpo; o BOA deles fica indefinido e sai da mediana,
como manda a convenção do artigo [1] (ver abaixo). A faixa clear 0–10% passa de 162 para 97 patches com BOA
definido: a estimativa feita sobre as contagens antigas foi confirmada pelas contagens novas (ver "Confirmação
depois da correção").

O n = 57 do projeto anterior não foi reproduzido. Com o código antigo (`stratified_analysis` em
`src/cloudsen12/evaluation/evaluator.py` [6]) aplicado a este teste, todas as variantes testadas dão outros
valores:

| Variante | n em clear 0–10% |
|---|---|
| Agrupando pela porcentagem de céu limpo, só patches com BOA definido (`dropna`) | 162 |
| Porcentagem sobre o patch inteiro em vez dos pixels anotados | 162 (não há pixel sem anotação) |
| Faixa fechada em 10% (`pd.cut`, `right=True`) | 162 (nenhum patch na borda) |
| Agrupando pela cobertura de nuvem, o padrão `stratify_by="cloud_cover"` | 87 |
| Agrupando pela porcentagem de pixels inválidos (nuvem + sombra) | 52 |
| Sem a faixa de preenchimento (recorte corrigido) | 97 |

O projeto anterior lia o canvas inteiro de 512 sem recorte, portanto com as duas bordas de preenchimento. A
diferença que resta deve vir de outro detalhe da execução antiga (dados, versão do dataset ou argumentos da
chamada), que não está registrado no repositório. Fica em aberto.

### Convenção do BOA usada nas figuras

O artigo do CloudSEN12 reporta o BOA como a mediana sobre os patches: *"The BOA value is computed as the median
BOA across all IPs"* (nota da Tabela 6) [1]. O texto não diz explicitamente que patches sem a classe ficam de fora
da mediana do BOA; diz isso só para PA e UA. A regra usada aqui (mediana sobre os patches em que o BOA está
definido, isto é, com a classe e o complemento) foi conferida empiricamente: `eval/report.cloudsen12_table`, com
essa regra, reproduz as 22 linhas publicadas da Tabela 6 a partir das métricas por patch dos autores.

### Referências

1. Aybar, C. et al. CloudSEN12, a global dataset for semantic understanding of cloud and cloud shadow in
   Sentinel-2. *Scientific Data* 9, 782 (2022). https://doi.org/10.1038/s41597-022-01878-2
2. Aybar, C. et al. CloudSEN12+: The largest dataset of expert-labeled pixels for cloud and cloud shadow detection
   in Sentinel-2. *Data in Brief* 56, 110852 (2024). https://doi.org/10.1016/j.dib.2024.110852
3. TACO Foundation. CloudSEN12 dataset card (README), seção sobre `real_proj_shape` e preenchimento.
   https://huggingface.co/datasets/tacofoundation/cloudsen12 (acesso em 15/11/2026).
4. Wright, N., Duncan, J. M. A., Callow, J. N., Thompson, S. E., George, R. J. Training sensor-agnostic deep
   learning models for remote sensing: Achieving state-of-the-art cloud and cloud shadow identification with
   OmniCloudMask. *Remote Sensing of Environment* 322, 114694 (2025). https://doi.org/10.1016/j.rse.2025.114694 —
   código do pacote `omnicloudmask` 1.7.0 (https://pypi.org/project/omnicloudmask/1.7.0/,
   https://github.com/DPIRD-DMA/OmniCloudMask): `cloud_mask.predict_from_array` e `raster_utils.mask_prediction`.
5. Projeto anterior, `notebooks/baseline.ipynb`, células 6 e 10 (figuras do canvas de 512):
   https://github.com/ArthurrCr/cloud-segmentation/blob/f9fc8a301f7c1fe2e22e4e59e737d29276838188/notebooks/baseline.ipynb
6. Projeto anterior, `stratified_analysis`:
   https://github.com/ArthurrCr/cloud-segmentation/blob/b18190359c5a7c4737f5f6c62223439a8af35417/src/cloudsen12/evaluation/evaluator.py

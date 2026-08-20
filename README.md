# dl-vigilancia-transformer

Deteccion de eventos sospechosos (robo, hurto, asalto) en videos de
vigilancia mediante un CNN feature extractor + Transformer Encoder,
adaptando el ejemplo oficial de Keras "Video Classification with
Transformers" a un problema de seguridad urbana.

Entrenado sobre UCF-Crime (Sultani et al., CVPR 2018) y evaluado/
adaptado a un dataset de videos de robos en camaras de noticieros
peruanos, para medir y mitigar el domain gap entre ambos contextos.

## Requisitos

- Windows 10/11, PowerShell
- Python 3.10+
- GPU NVIDIA con drivers CUDA (opcional pero recomendado; el proyecto
  corre igual en CPU, solo mas lento)

## 1. Instalacion

Todos los comandos se ejecutan **desde la raiz del repo**.

```powershell
.\install_torch.ps1
```

Este script crea el entorno virtual `.venv`, detecta si hay GPU NVIDIA
disponible e instala PyTorch con CUDA (build cu128) o CPU segun
corresponda, luego instala el resto de dependencias desde
`requirements.txt` (que incluye `keras>=3.0`, configurado para usar
PyTorch como backend). Verifica al final con:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Estructura del repositorio

```bash
dl-vigilancia-transformer/
├── data/
│ ├── raw/
│ │ ├── ucf_crime/<Clase>/.mp4 # NO se sube al repo, descargar aparte (ver seccion 3)
│ │ ├── peru_robos/Robbery|Normal/.mp4
│ │ ├── mini_dataset/ # prueba rapida del pipeline
│ │ └── UCF_Crimes-Train-Test-Split/ # split oficial, incluido en el repo (12KB)
│ └── processed/
│ ├── train/, val/ # secuencias .npy generadas (NO se suben)
│ ├── features/ # cache de features CNN (NO se sube)
│ └── manifest.csv # SI se sube, registro de que secuencia es cada video
├── models/<experimento>/
│ ├── config.json # hiperparametros exactos usados
│ ├── best_model.weights.h5 # mejor checkpoint segun val_accuracy
│ ├── last_checkpoint.weights.h5
│ ├── history.csv # metricas por epoca
│ ├── training_curves.png # loss/accuracy train vs val
│ ├── confusion_matrix.png
│ └── metrics.json
├── configs/default.yaml # configuracion activa (clases, hiperparametros)
├── scripts/
│ ├── extract_ucf_subset.py # extrae solo las clases objetivo de los zips de UCF-Crime
│ ├── augment_train.py # genera copias aumentadas del train set
│ └── list_holdout_videos.py # reproduce el split peru finetune/holdout
├── 01_preprocess.py # videos crudos -> secuencias .npy + manifest
├── 02_train.py # entrena un experimento (features CNN + Transformer)
├── 03_evaluate.py # evalua un modelo entrenado contra el dataset peruano
├── 04_finetune_peru.py # domain adaptation: fine-tuning con mitad de Peru
├── 05_live_demo.py # deteccion en vivo (webcam o archivo) con OpenCV
├── install_torch.ps1
└── requirements.txt

```


**Todos los scripts numerados deben correrse desde la raiz del repo**
(usan rutas relativas tipo `data/...`, `configs/default.yaml`).

## 3. Obtener y organizar los datasets

Ninguno de los videos crudos va en el repo (`.gitignore` los excluye).
Hay que descargarlos y organizarlos manualmente antes de preprocesar.

### 3.1 UCF-Crime (clases: Robbery, Burglary, Shoplifting, Stealing, Normal)

Fuente oficial: https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/

1. Descargar `UCF_Crimes-Train-Test-Split.zip` (12KB)  ya viene incluido
   en el repo en `data/raw/UCF_Crimes-Train-Test-Split/`, no hace falta
   volver a bajarlo salvo que se borre.
2. Descargar los 4 zips `Anomaly-Videos-Part-1..4.zip` (~23GB en total) y
   `Normal_Videos_for_Event_Recognition.zip` (1GB) desde el Dropbox
   oficial, uno a la vez, en `data/raw/downloads/`.
3. Por cada zip descargado, correr:
```powershell
   python scripts\extract_ucf_subset.py
```
   Este script lee el split oficial (`train_001.txt` + `test_001.txt`
   de `Action_Regnition_splits/`) y extrae **solo** los videos de
   nuestras 5 clases objetivo hacia `data/raw/ucf_crime/<Clase>/`,
   ignorando el resto (Abuse, Arrest, Arson, etc.). No hace falta
   extraer los zips completos ni tenerlos todos al mismo tiempo en
   disco se puede borrar cada zip despues de procesarlo.

   Mapeo confirmado de que categoria esta en que Part (verificado
   directamente, no asumido):
   - Part-1: Abuse, Arrest, Arson, Assault
   - **Part-2: Burglary**, Explosion, Fighting
   - **Part-3: RoadAccidents, Robbery**, Shooting
   - **Part-4: Shoplifting, Stealing**, Vandalism
   - `Normal_Videos_for_Event_Recognition.zip`: Normal

4. Al terminar, cada carpeta en `data/raw/ucf_crime/<Clase>/` debe
   tener exactamente 50 videos (Robbery, Burglary, Shoplifting,
   Stealing, Normal).

### 3.2 Dataset peruano (clases: Robbery, Normal  solo para evaluacion/fine-tuning)

Fuente: https://www.kaggle.com/datasets/michaelmirandaxd/videos-de-robos-en-el-per

Descargar y copiar manualmente:
- carpeta `datafi/` (videos de robo) → `data/raw/peru_robos/Robbery/`
- carpeta `datano/` (videos normales) → `data/raw/peru_robos/Normal/`

Verificar:
```powershell
(Get-ChildItem data\raw\peru_robos\Robbery -Filter *.mp4).Count   # debe dar 100
(Get-ChildItem data\raw\peru_robos\Normal -Filter *.mp4).Count    # debe dar 100
```

**Importante:** este dataset nunca se usa en el entrenamiento principal
(`02_train.py`) solo en `03_evaluate.py` (evaluacion zero-shot) y
`04_finetune_peru.py` (domain adaptation, usando solo la mitad).

### 3.3 Agregar mas videos a UCF-Crime en el futuro (opcional)

Se pueden copiar `.mp4` adicionales directamente a
`data/raw/ucf_crime/<Clase>/` (cualquier nombre de archivo sirve,
mientras sea unico dentro de la carpeta para evitar colisiones). Los
videos que no formen parte del split oficial de UCF-Crime se asignan
automaticamente a `train` (nunca a `val`, para no contaminar la
validacion oficial).

## 4. Preprocesamiento

Convierte los `.mp4` en secuencias de frames de longitud fija, usando
el split oficial train/val de UCF-Crime, y arma el manifest:

```powershell
python 01_preprocess.py
```

Salida esperada: 190 secuencias en `train`, 60 en `val` (38/12 por
clase), y si `data/raw/peru_robos/` esta poblado, 200 mas en
`peru_eval` (100 Robbery + 100 Normal).

### Data augmentation (opcional)

Genera copias con flip horizontal + variacion de brillo/contraste
sobre el train set:

```powershell
python scripts\augment_train.py
```

> Nota de resultados: en nuestros experimentos, el augmentation
> **no mejoro** el desempeno (ver seccion 6), se documenta el script
> por completitud metodologica, no porque sea parte del pipeline
> recomendado final.

## 5. Entrenamiento

```powershell
python 02_train.py --config configs/default.yaml --experiment nombre_del_experimento
```

Cada experimento crea su propia carpeta en `models/<nombre>/`, con su
`config.json`, checkpoints, curvas y metricas no se pisan entre si.

Para reanudar un entrenamiento pausado:
```powershell
python 02_train.py --config configs/default.yaml --experiment nombre_del_experimento --resume
```

### Como correr un experimento nuevo con datos distintos (procedimiento completo)

Si se cambia algo que afecta el preprocesamiento (`sequence_length`,
`img_size`, o se agrega/quita augmentation), hay que regenerar los
datos derivados **antes** de entrenar, porque `01_preprocess.py`
sobrescribe los `.npy` con el nuevo formato pero la cache de features
(`data/processed/features/`) no se invalida sola:

```powershell
# 1) Editar configs/default.yaml con los nuevos hiperparametros

# 2) Borrar los derivados desactualizados
Remove-Item -Recurse -Force data\processed\train, data\processed\val, data\processed\features

# 3) Regenerar secuencias
python 01_preprocess.py

# 4) (opcional) Regenerar augmentation si se va a usar
python scripts\augment_train.py

# 5) Entrenar el nuevo experimento
python 02_train.py --config configs/default.yaml --experiment nombre_experimento_nuevo
```

Si solo se cambian hiperparametros del modelo (`dense_dim`,
`num_heads`, `dropout`, `learning_rate`, `batch_size`, `epochs`), que
no afectan el preprocesamiento **no hace falta borrar nada**, se
puede ir directo al paso 5 con un yaml distinto:

```powershell
python 02_train.py --config configs/otro_experimento.yaml --experiment nombre_experimento_nuevo
```

## 6. Resumen de experimentos realizados

| Experimento | Train | Capacidad | Augmentation | seq_len | Mejor val_accuracy |
|---|---|---|---|---|---|
| exp01_baseline | 190 | Full (512/4) | No | 20 | **0.633** |
| exp02_augmented_earlystop | 570 | Reducida (256/2) | Flip+brillo | 20 | 0.583 |
| exp03_augmented_fullcapacity | 570 | Full (512/4) | Flip+brillo | 20 | 0.567 |
| exp04_augmented_noflip | 570 | Full (512/4) | Solo brillo | 20 | 0.567 |
| exp05_seqlen32 | 190 | Full (512/4) | No | 32 | 0.550 |

Conclusion: ni el augmentation ni un `sequence_length` mayor superaron
al baseline, el techo esta en la cantidad de videos unicos por clase
(38 en train), no en la arquitectura. `exp01_baseline` es el mejor
modelo entrenado solo con UCF-Crime.

## 7. Evaluacion cross-domain (UCF-Crime → Peru)

```powershell
python 03_evaluate.py --experiment exp01_baseline
```

Evalua el modelo (zero-shot, nunca vio datos peruanos) contra los 200
videos de `peru_eval`. Guarda matriz de confusion, desglose de
confusiones por clase y metricas en `models/exp01_baseline/peru_eval/`.

Resultado obtenido: recall Robbery = 0.190, recall Normal = 0.410,
accuracy global = 0.300, evidencia clara de domain gap.

## 8. Domain adaptation (fine-tuning con datos peruanos)

```powershell
python 04_finetune_peru.py --base_experiment exp01_baseline --experiment exp06_domain_adapted
```

Parte de los pesos de `exp01_baseline`, continua el entrenamiento
agregando 50 Robbery + 50 Normal peruanos (mitad del dataset), y
evalua contra los 100 restantes (holdout, nunca visto ni en train ni
en fine-tuning).

Resultado obtenido: recall Robbery 0.190 → **0.600**, recall Normal
0.410 → **0.820**, accuracy 0.300 → **0.710**. A cambio, el modelo
pierde algo de precision en las clases originales de UCF-Crime
(catastrophic forgetting parcial, documentado en `training_curves.png`
de ese experimento).

Para ver exactamente que videos peruanos quedaron en el holdout
(reproducible, seed fija):
```powershell
python scripts\list_holdout_videos.py
```

## 9. Demo en vivo

```powershell
python 05_live_demo.py --experiment exp06_domain_adapted --source data\raw\peru_robos\Robbery\FIvideo-XXX.mp4 --save reports\figures\demo_output.mp4
```

`--source 0` usa la webcam (o `1`, `2`... para camaras externas).
`--save` es opcional, graba la salida a un `.mp4`. Se recomienda usar
clips del holdout peruano (paso 8) para que la demo sea consistente
con las metricas reportadas, no se recomienda improvisar frente a
la webcam, ya que el modelo nunca vio ese tipo de encuadre (ver
discusion de domain gap en el informe).

## Autores

- Rivas Huanca Diego Raul
- Guerra Chura Joan Leonard

## Referencias

- Sultani, W., Chen, C., & Shah, M. (2018). Real-world Anomaly
  Detection in Surveillance Videos. CVPR.
- Keras Team. Video Classification with Transformers.
  https://keras.io/examples/vision/video_transformers/
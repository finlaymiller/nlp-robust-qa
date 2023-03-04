# qa

Project for CSCI6908 NLP with Deep Learning

## Info

### Models

distilbert-base-uncased
distilbert-base-cased-distilled-squad
machine2049/distilbert-base-uncased-finetuned-duorc_distilbert
gsgoncalves/distilbert-base-uncased-race

### Datasets

squad
duorc
race

## Results

| Model                                                          | Test  | Fine-tuned | EM     | F1     |
|----------------------------------------------------------------|-------|------------|--------|--------|
| distilbert-base-uncased                                        | squad | No         | 0.293  | 8.144  |
| distilbert-base-uncased                                        | duorc | No         | 0.061  | 2.536  |
| distilbert-base-cased-distilled-squad                          | squad | No         | 79.650 | 87.025 |
| distilbert-base-cased-distilled-squad                          | duorc | No         | 52.712 | 65.747 |
| machine2049/distilbert-base-uncased-finetuned-duorc_distilbert | squad | No         | 52.705 | 62.143 |
| machine2049/distilbert-base-uncased-finetuned-duorc_distilbert | duorc | No         | 43.067 | 55.189 |
| distilbert-base-uncased                                        | squad | squad      | 62.743 | 73.184 |
| distilbert-base-uncased                                        | duorc | duorc      | 35.884 | 44.896 |
| distilbert-base-uncased                                        | duorc | squad      | 37.234 | 49.382 |
| distilbert-base-uncased                                        | squad | duorc      | 33.330 | 42.270 |


# Blender RAG project

## Contents

## Purpose of the project

## Deployment

## Limitations and solutions

## Main configuration

| Configuration setting | Value               | Comments                                                          |
| --------------------- | ------------------- | ----------------------------------------------------------------- |
| TEXT_FILE             | BlenderManual.txt   | Source for my database _(1)_                                      |
| EMBEDDING_MODEL       | intfloat/e5-base-v2 | Setting from best cfg _(2)_                                       |
| CHUNK_SIZE            | 1000                | Setting from best cfg _(2)_                                       |
| CHUNK_OVERLAP         | 100                 | Setting from best cfg _(2)_                                       |
| MIN_CHUNK_LEN         | 80                  | Empirical setting based on specifics of the documentation _(3)_   |
| BATCH_SIZE            | 5000                | Batching for DB upload _(4)_                                      |
| N_RESULTS             | 3                   | Setting from best cfg _(2)_                                       |
| DIST_THRESHOLD        | 0.8                 | Threshold used to limit text generation model functionality _(5)_ |
| LLM_MODEL             | openai/gpt-oss-120b | Text generation model _(6)_                                       |

### _Comments on the configuration_

## Repository structure

## Cloning/Forking setup

## Important notes

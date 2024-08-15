# MEDSQ: A framework for promoting the reliability of LLMs as assistants in medical education

## Dataset Sources
**Our datasets is constructed from two uni-modal resources: [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) and [Chest ImaGenome](https://physionet.org/content/chest-imagenome/1.0.0/).**

![source](https://github.com/JaneGovan/MEDSQ/blob/main/images/source.png)

## Statistics
### Abnormal zones
**The proportion of each abnormal zones in the entire datasets.**

![abnormal_zones](https://github.com/JaneGovan/MEDSQ/blob/main/images/abnormal_zones.png)

### Reponses
**The average number of words per round of responses in each template.**

![length_tokens](https://github.com/JaneGovan/MEDSQ/blob/main/images/length_tokens.png)

## Dataset Cases
### [EduDiag](https://github.com/JaneGovan/MEDSQ/blob/main/data/EduDiag.json)
#### English

![EduDiag_case_en](https://github.com/JaneGovan/MEDSQ/blob/main/images/EduDiag_case_en.png)

#### Chinese

![EduDiag_case_zh](https://github.com/JaneGovan/MEDSQ/blob/main/images/EduDiag_case_zh.png)

### [ScqTest](https://github.com/JaneGovan/MEDSQ/blob/main/data/ScqTest.json)
#### English

![ScqTest_case_en](https://github.com/JaneGovan/MEDSQ/blob/main/images/ScqTest_case_en.png)

#### Chinese

![ScqTest_case_zh](https://github.com/JaneGovan/MEDSQ/blob/main/images/ScqTest_case_zh.png)

## Dataset Details
**Each data in EduDiag contains `image`, `report_en`, `qa_en`, `report_zh`, `qa_zh`. `image` records the information contained in the patient's chest X-ray, `image_path` indicates the path of the image, `reason_for_exam` contains the patient's medical history and the purpose of the examination, `bbox` lists all anatomical locations with abnormalities and uses `focuses` to indicate specific abnormalities, and the remaining fields are directly derived from MIMIC-CXR. `report_en` and `report_zh` are cleaned English and Chinese medical reports respectively. `qa_en` and `qa_zh` contain multi-round `question` and `answer` of bilingual templates.**
```python
{
  "image": {
    "image_id": "19d2573b-bbbb5192-d992c5a2-7b72f28b-b6182646",
    "image_path": "img/19d2573b-bbbb5192-d992c5a2-7b72f28b-b6182646.jpg",
    "viewpoint": "AP",
    "patient_id": 19422157,
    "study_id": 53040876,
    "gender": "F",
    "patient_age": "40-50",
    "reason_for_exam": "A woman with severe upper abdominal pain s/p endoscopy.  // evaluate for free air.",
    "bbox": [
      {
        "bbox_name": "left lower lung zone",
        "original_x": 1364,
        "original_y": 1882,
        "original_width": 777,
        "original_height": 723,
        "x": 119,
        "y": 138,
        "width": 57,
        "height": 53,
        "focuses": [
          "atelectasis",
          ...
        ]
      },
      ...
    ]
  },
  "report_en": "...",
  "qa_en": [
    [
      {
        "Question": "...",
        "Answer": "..."
      },
      ...
    ],
    ...
  ],
  "report_zh": "...",
  "qa_zh": [...]
}
```


**Each data of ScqTest contains `image`, `report_en`, `test_en`, `report_zh`, `test_zh`. `image` records the information contained in the patient's chest X-ray in the same way. `test_en` and `test_zh` are English and Chinese single-choice question banks.**
```python
{
  "image": {...},
  "report_en": "...",
  "test_en": [
    [
      {
          "Question": "...",
          "A": "...",
          "B": "...",
          "C": "...",
          "D": "...",
          "GT": "D"
      },
      ...
    ],
    ...
  ],
  "report_zh": "...",
  "test_zh": [...]
}
```
## Inference

## Continuous Fine-tuning

## Comparsion

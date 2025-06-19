# UNLP 2025 Shared Task on Detecting Social Media Manipulation

The Fourth Ukrainian NLP Workshop (UNLP 2025) organizes a Shared Task on Detecting Social Media Manipulation. This Shared Task aims to challenge and assess AI capabilities to detect and classify manipulation, laying the groundwork for progress in cybersecurity and the identification of disinformation within the context of Ukraine.

The task was held from January 15 till Apr 4, 2025. See the results and stay tuned for the full report on the shared task to be published at UNLP 2025.

The Kaggle environments remain open for further experimentation, though any submissions after the official deadline will be considered outside of the UNLP 2025 Shared Task.
The competition featured two tracks:
- [Technique Classification Kaggle competition](https://www.kaggle.com/t/f40f491a48b841ab938275c169d57075 )
- [Span Identification Kaggle competition](https://www.kaggle.com/t/d633d1fa08cb472598e5ae3772ece142)


Join the discussions in Discord via https://discord.gg/DYNnWaZD4a.

## Task Description

In this shared task, your goal is to build a model capable of identifying manipulation techniques in Ukrainian social media content (specifically, Telegram). In this context, “manipulation” refers to the presence of specific rhetorical or stylistic techniques aimed to influence the audience without providing clear factual support.

There are two tracks in this shared task:
- **Subtask 1 (Technique classification):** given the text of a post, identify which manipulation techniques are used, if any. This is a multilabel classification problem; a single post can contain multiple techniques.

- **Subtask 2 (Span identification):** given the text of a post, identify the specific spans of manipulative text, regardless of the manipulation technique. This is a binary token-level classification task, focusing on pinpointing exactly where the manipulative content occurs.

## Data

The dataset was provided by the [Texty.org.ua](https://texty.org.ua/) team. It consists of Ukrainian Telegram posts annotated for the presence of ten manipulation techniques. The annotation was performed by experienced journalists, analysts, and media professionals. Detailed explanations and examples of manipulation techniques are available in [data/techniques-en.md](./data/techniques-en.md).

The dataset was split into train, private and public test sets. Importantly, the train/public/private splits remained identical for both competition tracks to prevent any potential data leakage between them. The split is as follows:
- Training set: 3822 samples
- Private test set: 3824 samples
- Public test set: 1911 samples
  
The two main dirs for the two tracks are:
- [data/span_detection](./data/span_detection)
- [data/techniques_classification](./data/techniques_classification)

## Limitations

To ensure fair and reproducible results:

1. You may not use any Telegram data.
2. You are allowed and encouraged to use other external data, but you must verify that the data license permits research use.
3. You should use only open-source models in your solution. Proprietary models are allowed only for data generation.
4. All code must be openly published for reproducibility (on GitHub, HF, etc.).

## Evaluation

- **Subtask 1 Technique Classification:** Macro-F1 (for multilabel classification)

- **Subtask 2 Span Identification:** Token-level F1 (for binary span detection)

## Results

The UNLP 2025 Shared Task on Detecting Social Media Manipulation is officially closed! 🙌

⭐ The winner in Technique Classification is Team GA, achieving the highest score of 0.49

⭐ The winner in Span Identification is also Team GA, beating other solutions with a score of 0.64
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/0c12dcb1-7d11-4b2d-8e59-885239f5a167" width="45%" />
    <img src="https://github.com/user-attachments/assets/dffc0b7a-90e6-4c33-abc8-2a052517ed38" width="45%" />
</div>

Full report on the shared task — TBD.

## Publication

Participants in the shared task are invited to submit a paper to the UNLP 2025 workshop. Please see the [UNLP website](https://unlp.org.ua/call-for-papers/) for details. Accepted papers will appear in the ACL anthology and will be presented at a session of UNLP 2025 specially dedicated to the Shared Task.

Submitting a paper is not mandatory for participating in the Shared Task.

## Important Dates

- **December 20, 2024** — Shared task announcement  
- **January 15, 2024** — Release of train data  
- **January 27, 2024** — Second call for participation
- **March 23, 2025** — Registration deadline
- **March 31 (11:58 PM GMT +02:00), 2025** — Final submission of system responses  
- **April 4, 2025** — Results of the Shared Task announced and release of test data  
- **April 20, 2025** — Shared Task paper due  
- **May 19, 2025** — Notification of acceptance  
- **June 2, 2025** — Camera-ready Shared Task papers due  
- **July 31 or August 1, 2025** — Workshop date

## Contacts

Email: [kvrware@gmail.com](mailto:kvrware@gmail.com), [volodymyr.sydorskyi@gmail.com](mailto:volodymyr.sydorskyi@gmail.com), [romanyshyn.n@ucu.edu.ua](mailto:romanyshyn.n@ucu.edu.ua), [oleksiy.syvokon@gmail.com](mailto:oleksiy.syvokon@gmail.com)

## License

This repository’s content is licensed under the 
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
See the [LICENSE](LICENSE) file for the full text.

## References

[UNLP 2025 Call for Papers](https://unlp.org.ua/call-for-papers/)



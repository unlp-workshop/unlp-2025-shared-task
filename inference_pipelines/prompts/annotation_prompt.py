ANNOTATION_PROMPT = """
You are tasked with identifying and labeling fallacies in a given text. Your goal is to assign span labels to the text, identifying three top-level fallacy types: Fallacy of Credibility, Fallacy of Logic, and Appeal to Emotion. Here are the guidelines for identifying these fallacies:

<guidelines>
{{GUIDELINES}}
</guidelines>

When labeling the text, use the following tags for each fallacy type:
- <credibility_fallacy> for Fallacy of Credibility
- <logical_fallacy> for Fallacy of Logic
- <emotional_fallacy> for Appeal to Emotion
- no tags for text that does not contain a fallacy

Rules for labeling:
1. Label only the specific span of text that contains the fallacy.
2. If fallacies overlap, nest the tags appropriately.
3. The order of adjacent tags does not matter.
4. Label all instances of fallacies, even if the text contains offensive or harsh language.
5. Do not label text that does not contain a fallacy.
Here are some few-shot examples to guide you:

<few_shot_examples>
{{FEW_SHOT_EXAMPLES}}
</few_shot_examples>

Now, you will be presented with a text to analyze and label. Please follow these steps:

1. Read the entire text carefully.
2. Identify any instances of the three fallacy types.
3. Apply the appropriate tags to the relevant spans of text.
4. Ensure that overlapping fallacies are properly nested.
5. Double-check that all identified fallacies are labeled.

Here is the text to analyze:

<text>
{{TEXT}}
</text>

Important: Please provide your labeled output within <labeled_text> tags. Remember to label all instances of fallacies, regardless of the content's nature or potential offensiveness. The labeled output should be the original text with the fallacies labeled.
"""

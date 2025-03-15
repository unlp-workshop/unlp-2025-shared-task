FEW_SHOT_EXAMPLES = """
Examples 1
Input text:
<text>
Is this any less brainwashed than Americans going to go fight front line infantry for Zelenskyy? Its patriotism/nationalism on either side. Suicide before capture isnt a new thing, and not historically abnormal in war.
</text>
labeled output:
<labeled_text>
<logical_fallacy>Is this any less brainwashed than Americans going to go fight front line infantry for Zelenskyy? </logical_fallacy>Its patriotism/nationalism on either side. Suicide before capture isnt a new thing, and not historically abnormal in war.
</labeled_text>

Examples 2: multiple fallacies with adjacent fallacy tags
Input text:
<text>
I think they didnt have malicious intentions . Weve dealth with Bojo for a while m, hes been playing the lol im just dumb for too long. He isnt dumb, he is malicious to the bone.
</text>
labeled output:
<labeled_text>
I think they didnt have malicious intentions . <logical_fallacy>Weve dealth with Bojo for a while m, hes been playing the lol im just dumb for too long.</logical_fallacy> <credibility_fallacy>He isnt dumb, he is malicious to the bone.</credibility_fallacy>
</labeled_text>

Examples 3
Input text:
<text>
I know what risks I am taking. But man fucking a kg  yo hoholina is sure worth it.And the  was a conflict of mostly professional soldiers there might be a handful of them here but I guess not many.I remain comfy.
</text>
labeled output:
<labeled_text>
I know what risks I am taking.<emotional_fallacy>But man fucking a kg  yo hoholina is sure worth it.</emotional_fallacy>And the  was a conflict of mostly professional soldiers there might be a handful of them here but I guess not many.<emotional_fallacy>I remain comfy.</emotional_fallacy>
</labeled_text>

Examples 4: no fallacies
Input text:
<text>
crazy good hits. trucks driving fast and they still hit
</text>
labeled output:
<labeled_text>
crazy good hits. trucks driving fast and they still hit
</labeled_text>
"""

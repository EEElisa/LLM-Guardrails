<div align= "center">
    <h1> <i>Let Them Down Easy!<i> Contextual Effects of LLM Guardrails on User Perceptions and Preferences</h1>
</div>

<div align="center">
    <p>
        <b>Authors:</b><br>
        Mingqian Zheng<sup>1</sup>, Wenjia Hu<sup>1,2</sup><sup>*</sup>, Patrick Zhao<sup>3</sup><br>
        <b>Motahhare Eslami</b><sup>1</sup>, <b>Jena D. Hwang</b><sup>4</sup><br>
        <b>Faeze Brahman</b><sup>4</sup><sup>†</sup>, <b>Carolyn Rosé</b><sup>1</sup><sup>†</sup>, <b>Maarten Sap</b><sup>1</sup><sup>†</sup><br>
        <small>
            <sup>1</sup>Carnegie Mellon University &nbsp;&nbsp;&nbsp; <sup>2</sup>Pareto.ai<br>
            <sup>3</sup>Simon Fraser University &nbsp;&nbsp;&nbsp; <sup>4</sup>Allen Institute for AI<br>
            <sup>*</sup>Work done as a student at Carnegie Mellon University<br>
            <sup>†</sup>Co-advising authors<br>
            <a href="mailto:mingqia2@andrew.cmu.edu">mingqia2@andrew.cmu.edu</a>
        </small>
    </p>
</div>

<div align="center">
<img src="figures/guardrail-teaser.png" width="750px">
</div>


<b>Abstract:</b> Current LLMs are trained to refuse potentially harmful input queries regardless of whether users actually had harmful intents, causing a tradeoff between safety and user experience. Through a study of 480 participants evaluating 3,840 query-response pairs, we examine how different refusal strategies affect user perceptions across varying motivations. Our findings reveal that response strategy largely shapes user experience, while actual user motivation has negligible impact. Partial compliance---providing general information without actionable details---emerges as the optimal strategy, reducing negative user perceptions by over 50\% to flat-out refusals. Complementing this, we analyze response patterns of 9 state-of-the-art LLMs and evaluate how 6 reward models score different refusal strategies, demonstrating that models rarely deploy partial compliance naturally and reward models currently undervalue it.  This work demonstrates that effective guardrails require focusing on crafting thoughtful refusals rather than detecting intent, offering a path toward AI safety mechanisms that ensure both safety and sustained user engagement.

<div align="center">
<img src="figures/flowchart.png" width="750px">
<p style="font-size: 0.9em; max-width: 750px; margin-top: 10px; text-align: center; font-style: italic;">
    Example user study flow for the chatbot interaction corresponding to safety category <i>Offensive Language</i> (top left). Participants select topics from a given list (middle) and read the given motivation (benign or malicious). The model's response strategy is determined by the experimental condition: in aligned settings, benign queries receive full compliance while malicious queries receive the assigned refusal strategy; in misaligned settings, this pattern is reversed. Participants immediately evaluate each response across multiple perception dimensions (right).
</p>
</div>


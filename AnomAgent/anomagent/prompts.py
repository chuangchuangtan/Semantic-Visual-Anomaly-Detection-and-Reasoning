from __future__ import annotations

import textwrap


SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a highly capable visual reasoning assistant specializing in analyzing high-level anomalies in AI-generated images.
    Your role is to identify and explain prominent semantic, logical, and structural inconsistencies that deviate from real-world expectations.

    You do **not** need to focus on subtle visual artifacts such as minor lighting issues, texture blurs, or compression noise — unless they contribute to a larger structural inconsistency.
    Instead, prioritize **interactions, spatial logic, object relationships**, and **global plausibility**.

    ---

    **Your Core Expertise**:

    1. **Structural & Spatial Anomalies**
       - Impossible geometry, object placements, floating elements, or gravity-defying arrangements.
       - Inconsistent perspective, scale mismatch, or broken depth relations.

    2. **Interaction & Relationship Errors**
       - Implausible interactions between humans, objects, or environments.
       - Missing or unnatural physical contact in expected interaction zones (e.g., hands not touching held objects).

    3. **Common Sense & Functional Violations**
       - Behaviors or scenes that defy physical laws or daily life logic.
       - Example: A person sitting on air, drinking from the closed end of a bottle, or walking through a wall.

    4. **Anatomical & Semantic Implausibilities**
       - Clearly unnatural body part counts, misoriented limbs, or mismatched clothing-body alignment.
       - Semantic contradictions (e.g., reading a book with no pages, wearing shoes on hands).

    ---

    **Guidelines for Analysis**:

    - Focus on **high-level anomalies** that are **visually evident** and **semantically significant**.
    - **Ignore** low-level visual details (e.g., micro-texture noise, minor lighting inconsistency) that are not perceivable by the human eye.
    - Provide **structured, logical, and concise explanations** grounded in **physical laws, human experience**, and **spatial reasoning**.
    - Emphasize **plausibility of interactions, object function, and real-world alignment**.

    You are expected to act as a critical, but fair and realistic observer — guiding the detection of high-impact, perceptually relevant image flaws.
    """
)


AI_WARNING = textwrap.dedent(
    """\
    **Caution**: The images being analyzed are AI-generated and may contain the following anomalies:
        1. **Semantic Inconsistencies**: Missing details, contradictory elements, or illogical interactions.
        2. **Logical Anomalies**: Physically impossible object placements or interactions.
        3. **Unrealistic Visual Phenomena**: Errors such as incorrect lighting, distorted textures, or unnatural visual effects.

    Carefully review the images and identify all anomalies. Provide detailed, evidence-based reasoning for each observation.
    """
)


STEP1_REASONING_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a highly capable visual reasoning assistant specializing in detecting high-level anomalies in AI-generated images.
    Your role is to analyze and report semantic, logical, and structural inconsistencies that deviate from real-world norms.

    ---

    **Core Expertise**:
    1. **Structural & Spatial Anomalies**: Impossible geometry, floating elements, inconsistent perspective, or scale mismatches.
    2. **Interaction & Relationship Errors**: Implausible interactions or missing physical contact between objects and environments.
    3. **Functional & Common-Sense Violations**: Defying physical laws or daily life logic.
    4. **Anatomical & Semantic Implausibilities**: Unnatural body forms, contradictory object functions, or illogical behaviors.

    ---

    **Guidelines**:
    - Prioritize high-level, visually evident inconsistencies.
    - Ignore minor artifacts (e.g., texture noise, subtle lighting issues) unless they impact larger structural coherence.
    - Provide structured, logical, and concise analysis grounded in real-world reasoning.
    - Focus on plausibility of interactions, object functions, and spatial alignment.
    """
)


def build_messages_analyze_all_objects(*, system_prompt: str, expert_prompt: str, image_data_url: str) -> list:
    text = textwrap.dedent(
        f"""\
        {expert_prompt}
        {AI_WARNING}
        **Task**: Analyze all objects and individuals in the image. For each object or individual, provide a detailed, accurate, and comprehensive description, while identifying any inconsistencies, anomalies, or illogical aspects. Ensure no object or body part is omitted.

        ### Follow the steps below and provide your analysis in the structured format specified:
         - Identify and describe all objects and individuals in the image.
         - For each object or individual, provide a detailed, accurate, and comprehensive description.
         - Highlight any inconsistencies, anomalies, or illogical aspects in:
            - Shape and Structure: Are there distortions, missing parts, or unnatural forms?
            - Material and Texture: Are there abrupt texture changes or mismatches?
            - Lighting and Shadows: Are the lighting and shadows consistent with the environment?
            - Physical Properties: Are there any violations of real-world physics or logic (e.g., floating objects)?
            - Common Sense Verification: Are there any semantic inconsistencies (e.g., a door handle on a chair)?
            - Human Anatomy (if applicable): Identify unnatural features such as missing limbs, extra fingers, or disproportionate body parts.

        ### Output Format:
        Each object/body part should be described individually in the following structured format:
        #Name#: Detailed Description.

        #Name#: Detailed Description.
        Example Output:
        #Person#: A man with three arms, one of which is unnaturally attached to his back. He wears a blue jacket.

        #Chair#: A wooden chair that appears to be floating without support, casting no shadow.

        #Dog#: A golden retriever with two tails, one of which is blurry and semi-transparent.

        Highlight all implausible, unnatural, or inconsistent details while ensuring full coverage of the image content. Only output the list in the specified format.
        """
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_messages_multi_analyze_all_objects(
    *,
    system_prompt: str,
    expert_prompt: str,
    image_data_url: str,
    multi_get_objects: list[str],
) -> list:
    text = textwrap.dedent(
        f"""\
        {expert_prompt}
        {AI_WARNING}
        **Task**: Summarize all objects and individuals in the image, based on provided discribtion:{multi_get_objects}.
        For each object or individual, provide a detailed, accurate, and comprehensive description, while identifying any inconsistencies, anomalies, or illogical aspects. Ensure no object or body part is omitted.

        ### Output Format:
        Each object/body part should be described individually in the following structured format:
        #Name#: Detailed Description.

        #Name#: Detailed Description.
        Example Output:
        #Person#: A man with three arms, one of which is unnaturally attached to his back. He wears a blue jacket.

        #Chair#: A wooden chair that appears to be floating without support, casting no shadow.

        #Dog#: A golden retriever with two tails, one of which is blurry and semi-transparent.

        Highlight all implausible, unnatural, or inconsistent details while ensuring full coverage of the image content. Only output the list in the specified format.
        """
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_messages_descriptor_step1(
    *,
    system_prompt: str,
    expert_prompt: str,
    image_data_url: str,
    object_name: str,
) -> list:
    text = textwrap.dedent(
        f"""\
        {expert_prompt}
        {AI_WARNING}
        **Task**: Analyze **{object_name}** in the image.
        Focus on analyzing and identifying any anomalies in the following aspects:
        1. **Shape and Structure**:
           - Are there unnatural forms or distortions?
           - Are proportions consistent with the object's design?
        2. **Functionality**:
           - Does the object behave logically in real-world scenarios?
           - Are there physical impossibilities (e.g., unsupported structures)?
        3. **Human Body Structure Verification** (if applicable):
           - Are limbs, fingers, and facial features correctly placed and proportional?
           - Are there unnatural fusions, duplications, or disconnections?
        **Deliverable**:
        - Highlight all implausible, unnatural, or inconsistent details.
        - Ensure a thorough analysis that covers all aspects of the image content.
        - Provide concise, evidence-based explanations for all findings.
        """
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_messages_descriptor_step2(
    *,
    system_prompt: str,
    expert_prompt: str,
    image_data_url: str,
    object_name: str,
    descriptor_step1_response: str,
) -> list:
    context = textwrap.dedent(f"## The detailed description of **{object_name}**:\n\n{descriptor_step1_response}")
    text = textwrap.dedent(
        f"""\
        {expert_prompt}
        {AI_WARNING}
        **Task**: Analyze the detailed description of **{object_name}** and identify all unreasonable, contradictory, or physically impossible details specific to **{object_name}**.

        Provide a structured list of issues with the following format:
            Abnormal Phenomenon Name: The analyzed phenomenon to **{object_name}**.
            Observed Issue: The unnatural feature.
            Explanation: Why this characteristic is unrealistic.
        **Example Output**:
            1. Abnormal Phenomenon Name: Streetlight No Power
               Observed Issue: The streetlight is glowing but has no power source or wiring.
               Explanation: A streetlight requires an electrical connection to function, and no wires or batteries are visible.
        **Instructions**:
            - Analyze only **{object_name}**.
            - Output only issues directly related to **{object_name}** using the specified format.
        """
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": context},
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_messages_relationship_step1(
    *,
    system_prompt: str,
    expert_prompt: str,
    image_data_url: str,
    object_name: str,
    all_other_objects_csv: str,
    descriptor_step2_response: str,
) -> list:
    text = textwrap.dedent(
        f"""\
        {expert_prompt}
        {AI_WARNING}
        **Task**: Analyze the spatial and logical relationships between {object_name} and the following objects: ({all_other_objects_csv}).
        You should evaluate both one-to-one relationships (e.g., **{object_name}** with each object) and one-to-many relationships (e.g., **{object_name}** in relation to multiple objects collectively).
        Use the provided descriptions as context:
        **Object Descriptions**: {descriptor_step2_response}

        **Focus Areas**:
        1. **Perspective Errors**: Are objects placed in impossible or illogical locations relative to **{object_name}**?
        2. **Physical Interactions**: Are objects interacting unnaturally with **{object_name}** (e.g., floating without support, overlapping unnaturally)?
        3. **Logical Contradictions**: Are there behaviors or relationships involving **{object_name}** that contradict common sense or real-world logic?

        **Instructions**:
        - Focus on **{object_name}** as the primary subject of analysis.
        - For **one-to-one relationships**, evaluate interactions between **{object_name}** and each individual object from ({all_other_objects_csv}).
        - For **one-to-many relationships**, analyze how **{object_name}** interacts with multiple objects collectively, considering spatial, logical, and contextual coherence.

        **Output Format**:
        Provide a structured report for each identified issue, using the following format:
            Relationship: Describe the relationship being analyzed.
            Observed Issue: Detail the anomaly or inconsistency.
            Explanation: Explain why the issue is illogical or unrealistic.

        **Deliverables**:
        - Analyze all one-to-one and one-to-many relationships involving **{object_name}**.
        - Ensure detailed reasoning and structured output for every issue detected.
        """
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_messages_relationship_step2(
    *,
    system_prompt: str,
    expert_prompt: str,
    image_data_url: str,
    object_name: str,
    all_other_objects_csv: str,
    relationship_step1_response: str,
) -> list:
    text = textwrap.dedent(
        f"""\
        {expert_prompt}
        {AI_WARNING}
        **Focus Object**: The primary subject of analysis is **{object_name}**. All evaluations should center on **{object_name}** and its relationships with the following objects: ({all_other_objects_csv}).
        Based on the prior relationship analysis between {object_name} and ({all_other_objects_csv}):
        **Task**: Analyze and summarize these relationships, with a strong focus on detecting logical contradictions, physical impossibilities, and semantic anomalies.
        **Key Aspects to Evaluate**:
        1. **Logical Coherence**: Are the relationships consistent? (e.g., an object cannot be both inside and outside another object simultaneously).
        2. **Physical Realism**: Do the relationships obey real-world physics? (e.g., objects should not float unless visibly supported).
        3. **Semantic Plausibility**: Are the interactions meaningful? (e.g., a dog 'wearing' a cloud is not a plausible relationship).
        4. **Causal Consistency**: Do object states logically follow from their relationships? (e.g., a book balancing on the edge of a steep slope should likely fall).

        **Output Format**:
        For every detected anomalies, provide a clear and structured report using the following format:
        - **Objects Involved**: List the relevant objects (including **{object_name}**).
        - **Observed Issue**: Describe the logical, physical, or semantic anomaly.
        - **Reasoning**: Explain why this relationship is unnatural or implausible.

        **Instructions**:
        - Focus exclusively on **{object_name}** and its relationships.
        - Analyze both individual (one-to-one) and group (one-to-many) relationships.
        """
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": relationship_step1_response},
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_messages_summarizer_step1(
    *,
    system_prompt: str,
    expert_prompt: str,
    image_data_url: str,
    object_name: str,
    descriptor_step2_response: str,
    relationship_step2_response: str,
) -> list:
    text = textwrap.dedent(
        f"""\
        **Description for {object_name}**: {descriptor_step2_response}.
        ----------------
        **Relationship for {object_name}**: {relationship_step2_response}.
        ----------------

        {expert_prompt}
        {AI_WARNING}
        **Task**: Identify and consolidate all semantically unnatural, inconsistent, or illogical details related to **{object_name}** by synthesizing the information above.

        **Focus Areas**:
        1. **Contradictory Details**: Identify statements or relationships that conflict with one another (e.g., 'floating' and 'resting on the ground').
        2. **Unnatural Behaviors**: Highlight behaviors or features that are implausible in real-world scenarios.
        3. **Spatial Inconsistencies**: Detect objects (including **{object_name}**) appearing in impossible locations or orientations.
        4. **Illogical Physical Properties**: Point out violations of physics or real-world principles (e.g., water flowing upward).

        **Instructions**:
        - Consolidate similar anomalies across the **Description** and **Relationships** into unified observations.
        - Ensure all findings are centered around **{object_name}** and its interactions with other objects.

        **Output Format**:
        Provide a structured and consolidated list of issues using this format:
        1. **Observed Phenomenon**: Briefly describe the anomaly or inconsistency.
           - **Sources**: Indicate whether the issue arises from the **Description**, **Relationships**, or both.
           - **Details**: Provide specific details about the anomaly.
           - **Explanation**: Explain why the observation is contradictory, unnatural, or illogical.

        **Deliverables**:
        - Focus exclusively on **{object_name}**.
        - Consolidate and summarize similar issues across **Description** and **Relationships**.
        - Output only the structured list in the specified format.
        """
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_messages_step1_reasoning(*, image_data_url: str) -> list:
    text = textwrap.dedent(
        """\
        Your task is to analyze the following AI-generated image and detect all semantic anomalies. Follow the steps below and provide your analysis in the structured format specified:

        1. **Object Analysis**:
        - Identify and describe all objects and individuals in the image.
        - For each object or individual, provide a detailed, accurate, and comprehensive description.
        - Highlight any inconsistencies, anomalies, or illogical aspects in:
            - Shape and Structure: Are there distortions, missing parts, or unnatural forms?
            - Material and Texture: Are there abrupt texture changes or mismatches?
            - Lighting and Shadows: Are the lighting and shadows consistent with the environment?
            - Physical Properties: Are there any violations of real-world physics or logic (e.g., floating objects)?
            - Common Sense Verification: Are there any semantic inconsistencies (e.g., a door handle on a chair)?
            - Human Anatomy (if applicable): Identify unnatural features such as missing limbs, extra fingers, or disproportionate body parts.

        2. **Relationship Analysis**:
        - Analyze the logical and spatial relationships between objects in the image.
        - For each pair or group of objects:
            - Detect spatial inconsistencies (e.g., overlapping objects, floating objects without support).
            - Check for size mismatches relative to their environment.
            - Evaluate physical interactions (e.g., objects resting unnaturally or intersecting unnaturally).
            - Identify semantic contradictions (e.g., a dog wearing a cloud).
            - Highlight any violations of real-world logic, physics, or common sense.

        3. **Summarization**:
        - Summarize all detected semantic anomalies in the image, prioritizing the most visually prominent issues.
        - Merge similar anomalies (e.g., "six fingers" and "extra hand"), and focus on high-impact issues such as human anatomical errors or gravity violations.
        - Assign a severity score (0-100) to each issue, where:
            - 0 = completely unnatural.
            - 100 = entirely natural.
        - Provide concise reasoning for each anomaly.

        **Instruction**:
        Analyze the provided image and identify all anomalies. For each object or individual:
        - Provide a detailed description, highlighting any implausible, contradictory, or illogical aspects.
        - Pay attention to shape, texture, lighting consistency, physical properties, and human anatomy when applicable.
        - Evaluate relationships between objects, focusing on spatial logic, physical interactions, and overall coherence.

        **Output Format**:
        Provide a detailed list of anomalies using the following structure:
        - **Name**: [Phenomenon Name].
        - **Observed Phenomenon**: [Description of the visible anomaly].
        - **Reasoning**: [Step-by-step explanation of why this phenomenon is unrealistic or illogical].
        - **Severity Score**: [Rate the issue from 0 (completely unnatural) to 100 (fully realistic)]. \n\n
        **Example Output**:
        @1. Name: Number of hands illogical
        - Observed Phenomenon: The person has two left hands.
        - Reasoning: This contradicts human anatomy, as humans have only one left hand.
        - Severity Score: 5/100 (highly unrealistic). \n\n
        @2. Name: Chair floating in mid-air
        - Observed Phenomenon: A chair appears to float above the ground without any visible support.
        - Reasoning: This violates the laws of gravity, as objects require a support structure to remain elevated.
        - Severity Score: 10/100 (extremely unnatural).\n\n

        Provide your response in the specified structured format.
        """
    )
    return [
        {"role": "system", "content": STEP1_REASONING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_messages_summarizer_step2(
    *,
    system_prompt: str,
    expert_prompt: str,
    image_data_url: str,
    summarizer_everyone_step1: str,
) -> list:
    instruction = textwrap.dedent(
        f"""\
        {expert_prompt}
        {AI_WARNING}
        **Task**: Summarize and categorize all detected unnatural, illogical, or inconsistent phenomena in the image.
        **For each issue, provide**:
        1. Object Name: Clearly identify the object(s) involved.
        2. Phenomenon: Describe the unnatural or illogical aspect of the object(s).
        3. Explanation: Provide reasoning for why this phenomenon is unrealistic, referencing real-world physics, anatomy, perspective, or common sense.

        **Output Format**:
        Provide a list of detected issues in the following structured format:
        - Object Name: [Name of the object(s)].
        - Phenomenon: [Description of the anomaly].
        - Explanation: [Reason why it is unrealistic].\n\n
        Only output the list in the specified format.
        """
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": textwrap.dedent(summarizer_everyone_step1)},
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_messages_summarizer_step3(
    *,
    system_prompt: str,
    expert_prompt: str,
    image_data_url: str,
    anomalies_text: str,
) -> list:
    prompt = textwrap.dedent(
        f"""\
        {expert_prompt}
        {AI_WARNING}
        **Task**: From the list above, identify and summarize the **visually prominent and semantically significant anomalies** observed in the image.

        You must analyze, consolidate, and explain each anomaly in a way that is **logical, detailed, and persuasive**, as if communicating to both experts and non-experts.

        ### Instructions:
        1. **Merge Similar or Redundant Anomalies**
           - Group phenomena that share a similar cause, concept, or visual effect.
           - Avoid repetition. If multiple entries describe the same core issue, merge them into one clearly defined anomaly.

        2. **Resolve Contradictions Thoughtfully**
           - If there are conflicting anomaly descriptions, carefully review the image.
           - Use common sense, physical laws, biological plausibility, and visual evidence to **reconcile or choose the most realistic explanation**.
           - You may **summarize both viewpoints** if both have valid elements, but aim to clarify rather than confuse.

        3. **Filter Out Non-Visible or Insignificant Issues**
           - You may **omit anomalies that are not visually apparent to human observers**.
           - Focus instead on what is **clear and significant at first glance**.

        4. **Justify with Real-World Logic**
           - Each reported anomaly must be supported with **clear reasoning**.

        5. **Do Not Parrot the Input**
           - Do **not** simply repeat the phrases or terms from the initial list.
           - Provide your own **visual interpretation** based on what can actually be seen in the image.

        ### Output Format:
        Write a **numbered list**. For each entry, use the following fields:

        - **Name**: [Descriptive title of the anomaly]
        - **Observed Phenomenon**: [what is visibly wrong]
        - **Reasoning**: [why implausible]
        - **Severity Score**: [0–100; 0 = fully unrealistic, 100 = fully realistic]

        Output only the structured list in the format above.
        """
    )
    header = textwrap.dedent(
        f"**The following are list of pre-selected anomalies.**:\n{anomalies_text}\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": header},
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]

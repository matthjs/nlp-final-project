class PreProcessor:
    """
    Preprocesses examples for Question Answering tasks.
    This preprocessor ONLY works properly
    for datasets that are formatted like the newsqa dataset ("lucadiliello/newsqa").
    """

    def __init__(self, tokenizer, max_length=512):
        """
        Initialize the PreProcessor.

        :param tokenizer: Tokenizer instance.
        :param max_length: Maximum length of input sequences. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess_function(self, examples):
        """
        Preprocess examples (Dataset object) for Question Answering tasks.

        :param examples: Examples to preprocess.
        :return: Processed inputs for Question Answering tasks.
        """
        # This function is modified from: https://huggingface.co/docs/transformers/tasks/question_answering
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        text_answers = examples["answers"]
        examples["answers"] = []
        labels = examples["labels"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            label = labels[i]
            start_char = label[0]["start"][0]
            end_char = label[0]["end"][0]
            sequence_ids = inputs.sequence_ids(i)

            # This stupid hack is needed because a lot of automatic evaluation in
            # huggingface for question answering tasks assumes squad like format.
            # Generate predictions in the required format.
            # This also means that whatever data you pass to preprocessor needs to have the same initial
            # format as "lucadiliello/newsqa"
            examples["answers"].append({"text": text_answers[i], "answer_start": [start_char]})

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
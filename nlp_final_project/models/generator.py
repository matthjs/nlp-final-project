from haystack.components.generators import HuggingFaceLocalGenerator

def fine_tuned_generator():
    # Using an open source language model.
    generator = HuggingFaceLocalGenerator(model="google/flan-t5-large",
                                               task="text2text-generation",
                                               generation_kwargs={
                                                   "max_new_tokens": 100,
                                                   "temperature": 0.9,
                                                   "do_sample": True
                                               })
    generator.warm_up()
    return generator
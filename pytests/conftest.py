from hypothesis import settings
import os

settings.register_profile("ci", settings(settings.get_profile("ci"), 
                                         max_examples=1000))
settings.register_profile("fast", max_examples=10)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))

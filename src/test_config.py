import google.cloud.aiplatform as aiplatform
import logging

logging.basicConfig(level=logging.DEBUG)
aiplatform.init(project="crafty-student-446923-n5", location="us-central1")
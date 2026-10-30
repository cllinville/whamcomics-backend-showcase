from .celery_app import celery

@celery.task
def generate_story_options_task(project_id: str):
    # TODO: call LLM to create outlines/loglines; write to DB
    return {"status": "ok", "project_id": project_id}

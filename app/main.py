from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .db import Base, engine
from . import models
from app.routers import projects, preview, uploads, story, checkout, characters, webhooks, book, admin, events

app = FastAPI(title="WhamComics Backend")
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(projects.router)
app.include_router(uploads.router)
app.include_router(story.router)
app.include_router(checkout.router)
app.include_router(characters.router)
app.include_router(webhooks.router)
app.include_router(preview.router)
app.include_router(book.router)
app.include_router(admin.router)
app.include_router(events.router)
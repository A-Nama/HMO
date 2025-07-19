import os
import json
import asyncio
import httpx
import traceback # Import the traceback module for detailed error logging
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from supabase import create_client, Client
import uuid
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# --- Environment Variable Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Pitch Deck PDF Generator & Viewer API (Public)",
    description="A public API to generate, save, and view pitch decks from audio or text.",
    version="6.7.0" # Added combined audio-to-slides feature endpoint
)

# --- CORS Configuration ---
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Client Initialization ---
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    raise ValueError("FATAL: Supabase credentials not found. Application cannot function.")

# --- Pydantic Models & Helper Functions ---
class PitchDeckSlide(BaseModel):
    title: str
    content: str

class PitchDeckContent(BaseModel):
    company_name: str
    tagline: str
    slides: List[PitchDeckSlide]

# Pydantic model for the text input of the new test endpoint
class IdeaTextInput(BaseModel):
    idea_text: str

def create_pitch_deck_prompt(idea_text: str) -> str:
    return f"""
    You are an expert startup consultant. Based on the user's idea: "{idea_text}", generate the content for a 10-slide pitch deck.
    Return a single, valid JSON object with the structure: {{"company_name": "...", "tagline": "...", "slides": [{{"title": "...", "content": "..."}}]}}.
    The 10 slides MUST be: 1. The Problem, 2. Our Solution, 3. Product/Service, 4. Market Size, 5. Business Model, 6. Go-to-Market Strategy, 7. Competitive Landscape, 8. Traction & Milestones, 9. Financial Projections, 10. The Ask.
    For the 'Traction & Milestones' slide, invent some plausible early achievements and future goals.
    """

async def _generate_deck_content(idea_text: str) -> dict:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = create_pitch_deck_prompt(idea_text)
        
        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }
        
        response = await model.generate_content_async(prompt, safety_settings=safety_settings)

        if response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason.name
            print(f"--- GEMINI CONTENT BLOCKED ---")
            print(f"Reason: {block_reason}")
            print("------------------------------")
            raise HTTPException(
                status_code=400, 
                detail=f"The content generation was blocked by the AI's safety filter. Reason: {block_reason}"
            )

        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print("--- FAILED TO PARSE JSON FROM GEMINI ---")
            print(f"Error: {e}")
            print("Problematic Text:")
            print(cleaned_text)
            print("-----------------------------------------")
            raise HTTPException(status_code=500, detail="Failed to parse content from AI. The response was not valid JSON.")

    except Exception as e:
        print("--- ERROR DURING GEMINI API CALL ---")
        traceback.print_exc()
        print("------------------------------------")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred with the AI model: {e}")


async def _generate_image_for_slide(slide_content: str, client: httpx.AsyncClient) -> bytes:
    if not HUGGING_FACE_API_KEY: 
        raise ValueError("HUGGING_FACE_API_KEY not found.")
        
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1-base"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    image_prompt = f"A professional, minimalist, abstract vector art representing: {slide_content}"
    
    async def get_fallback_image():
        fallback_url = "https://placehold.co/800x400/EEE/31343C?text=AI+Image+Failed"
        return (await client.get(fallback_url)).content

    try:
        response = await client.post(api_url, headers=headers, json={"inputs": image_prompt}, timeout=120.0)
        if response.status_code == 503: # Model is loading, retry once
            await asyncio.sleep(15)
            response = await client.post(api_url, headers=headers, json={"inputs": image_prompt}, timeout=120.0)
        
        if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
            return response.content
        else:
            print(f"Hugging Face returned non-image data. Status: {response.status_code}. Body: {response.text}")
            return await get_fallback_image()
    except Exception as e:
        print(f"Hugging Face image generation failed: {e}")
        return await get_fallback_image()

def _create_pitch_deck_pdf(deck_content: dict, images: list) -> io.BytesIO:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    styles = getSampleStyleSheet()

    # Title Page
    c.setFont("Helvetica-Bold", 36)
    c.drawCentredString(width / 2.0, height - 150, deck_content.get('company_name', 'Unnamed Company'))
    c.setFont("Helvetica", 18)
    c.drawCentredString(width / 2.0, height - 200, deck_content.get('tagline', ''))
    c.showPage()

    # Content Slides
    for i, slide in enumerate(deck_content.get('slides', [])):
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, height - 100, slide.get('title', 'Untitled Slide'))
        
        p = Paragraph(slide.get('content', 'No content available.'), styles['Normal'])
        p.wrapOn(c, width - 100, height - 450)
        p.drawOn(c, 50, height - 350)
        
        if i < len(images):
            try:
                image_reader = ImageReader(io.BytesIO(images[i]))
                c.drawImage(image_reader, 50, 80, width=width-100, height=200, preserveAspectRatio=True, anchor='n')
            except Exception as e:
                print(f"Could not draw image for slide {i}: {e}")

        c.showPage()

    c.save()
    buffer.seek(0)
    return buffer

# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
async def read_root():
    return {"message": "Pitch Deck API is up and running!"}

# --- NEW: Combined Feature Endpoint ---
@app.post("/audio-to-slides", tags=["Features"])
async def generate_slides_from_audio(audio_file: UploadFile = File(...)):
    """
    Accepts an audio file, transcribes it, and generates slide content.
    This combines transcription and content generation into one step.
    """
    print("--- Running Audio-to-Slides Feature ---")
    # Step 1: Transcribe Audio
    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
    
    try:
        print(f"Processing audio file: {audio_file.filename}")
        transcription_model = genai.GenerativeModel('gemini-1.5-flash')
        audio_bytes = await audio_file.read()
        audio_part = {"mime_type": audio_file.content_type, "data": audio_bytes}
        response = await transcription_model.generate_content_async(["Transcribe this startup idea.", audio_part])
        final_idea_text = response.text.strip()
        print("Transcription successful.")

        # Step 2: Generate Deck Content from transcription
        print(f"Generating content for idea: '{final_idea_text[:100]}...'")
        deck_content = await _generate_deck_content(final_idea_text)
        print("Content generation successful.")
        
        return JSONResponse(status_code=200, content=deck_content)

    except Exception as e:
        print("--- ERROR DURING AUDIO-TO-SLIDES FEATURE ---")
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


# --- Testing Endpoints ---

@app.post("/transcribe-test", tags=["Testing"])
async def transcribe_audio_test(audio_file: UploadFile = File(...)):
    """
    Accepts an audio file and returns the transcribed text.
    Use this to debug the audio processing and transcription step.
    """
    print("--- Running Transcription Test ---")
    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
    
    try:
        print(f"Processing audio file: {audio_file.filename} ({audio_file.content_type})")
        transcription_model = genai.GenerativeModel('gemini-1.5-flash')
        audio_bytes = await audio_file.read()
        audio_part = {"mime_type": audio_file.content_type, "data": audio_bytes}
        
        response = await transcription_model.generate_content_async(["Transcribe this startup idea.", audio_part])
        
        final_idea_text = response.text.strip()
        print(f"Transcription successful. Text: {final_idea_text}")
        
        return JSONResponse(
            status_code=200, 
            content={"transcribed_text": final_idea_text}
        )
        
    except Exception as e:
        print("--- ERROR DURING TRANSCRIPTION TEST ---")
        traceback.print_exc()
        print("---------------------------------------")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during transcription: {e}")

@app.post("/generate-content-test", tags=["Testing"])
async def generate_content_test(text_input: IdeaTextInput):
    """
    Accepts transcribed text and returns the generated 10-slide pitch deck content.
    Use this to debug the content generation step.
    """
    print("--- Running Content Generation Test ---")
    try:
        if not text_input.idea_text or len(text_input.idea_text) < 10:
            raise HTTPException(status_code=400, detail="Please provide a more detailed idea text.")

        print(f"Generating content for idea: '{text_input.idea_text[:100]}...'")
        deck_content = await _generate_deck_content(text_input.idea_text)
        print("Content generation successful.")
        
        return JSONResponse(status_code=200, content=deck_content)

    except Exception as e:
        print("--- ERROR DURING CONTENT GENERATION TEST ---")
        traceback.print_exc()
        print("--------------------------------------------")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during content generation: {e}")


# --- Main Application Endpoint ---
@app.post("/decks", tags=["Pitch Decks"])
async def generate_and_save_deck(
    audio_file: Optional[UploadFile] = File(None),
    idea_text: Optional[str] = Form(None)
):
    has_audio = audio_file and audio_file.filename

    if not has_audio and not idea_text:
        raise HTTPException(status_code=400, detail="Please provide either an audio file or typed text.")
    if has_audio and idea_text:
        raise HTTPException(status_code=400, detail="Please provide either an audio file or typed text, not both.")

    try:
        final_idea_text = ""
        print("--- Starting New Deck Generation ---")
        if has_audio:
            print("Step 1: Processing audio file...")
            if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
                raise HTTPException(status_code=400, detail="Invalid file type for audio.")
            
            transcription_model = genai.GenerativeModel('gemini-1.5-flash')
            audio_bytes = await audio_file.read()
            audio_part = {"mime_type": audio_file.content_type, "data": audio_bytes}
            response = await transcription_model.generate_content_async(["Transcribe this startup idea.", audio_part])
            final_idea_text = response.text.strip()
            print("Step 1: Audio transcribed successfully.")
            
        elif idea_text:
            print("Step 1: Processing text input...")
            if len(idea_text) < 20:
                raise HTTPException(status_code=400, detail="Text idea is too short. Please provide more detail.")
            final_idea_text = idea_text
            print("Step 1: Text input processed.")

        print("Step 2: Generating deck content with Gemini...")
        deck_content = await _generate_deck_content(final_idea_text)
        print("Step 2: Deck content generated successfully.")
        
        print("Step 3: Generating images for slides with Hugging Face...")
        async with httpx.AsyncClient() as client:
            image_tasks = [_generate_image_for_slide(slide['content'], client) for slide in deck_content['slides']]
            images = await asyncio.gather(*image_tasks)
        print("Step 3: All slide images generated.")

        print("Step 4: Creating PDF document...")
        pdf_buffer = _create_pitch_deck_pdf(deck_content, images)
        pdf_bytes = pdf_buffer.getvalue()
        print("Step 4: PDF document created.")
        
        print("Step 5: Uploading PDF to Supabase Storage...")
        file_path = f"public/{uuid.uuid4()}.pdf"
        supabase.storage.from_("pitch_decks").upload(file=pdf_bytes, path=file_path, file_options={"content-type": "application/pdf"})
        pdf_url = supabase.storage.from_("pitch_decks").get_public_url(file_path)
        print(f"Step 5: PDF uploaded successfully to {pdf_url}")

        print("Step 6: Saving deck metadata to Supabase database...")
        deck_to_save = {
            "company_name": deck_content.get("company_name"),
            "tagline": deck_content.get("tagline"),
            "original_idea": final_idea_text,
            "pdf_url": pdf_url,
            "storage_path": file_path
        }
        db_response = supabase.table("pitch_decks").insert(deck_to_save).execute()
        
        if not db_response.data:
             print("--- FAILED TO SAVE TO SUPABASE DB ---")
             print("Response:", db_response)
             raise HTTPException(status_code=500, detail="Failed to save deck information to the database.")
        
        print("Step 6: Deck metadata saved successfully.")
        print("--- Deck Generation Complete ---")
        return JSONResponse(status_code=201, content=db_response.data[0])

    except HTTPException as e:
        raise e
    except Exception as e:
        print("--- AN UNEXPECTED ERROR OCCURRED ---")
        traceback.print_exc()
        print("------------------------------------")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred. Check server logs for details.")


@app.get("/decks", tags=["Pitch Decks"])
async def get_all_decks():
    response = supabase.table("pitch_decks").select("*").order("created_at", desc=True).execute()
    return response.data

@app.delete("/decks/{deck_id}", tags=["Pitch Decks"])
async def delete_deck(deck_id: int):
    select_res = supabase.table("pitch_decks").select("storage_path").eq("id", deck_id).single().execute()
    if not select_res.data:
        raise HTTPException(status_code=404, detail="Deck not found.")
    
    storage_path = select_res.data['storage_path']
    supabase.storage.from_("pitch_decks").remove([storage_path])
    supabase.table("pitch_decks").delete().eq("id", deck_id).execute()
    
    return {"status": "success", "message": f"Deck {deck_id} deleted."}

# app.py
"""
songweaver-ai: Generate song lyrics and a simple melody from keywords.

This script uses a pretrained language model (GPT-2) from Hugging Face transformers
to create song lyrics based on a list of keywords. It also generates a simple melody
in a MIDI file using the midiutil library.

To install dependencies:
    pip install -r requirements.txt

Usage:
    python app.py love space "outer space" --lyrics-length 120 --melody-notes 32 --output-melody my_song.mid
"""

import argparse
import random
from transformers import pipeline, set_seed
from midiutil import MIDIFile


def generate_lyrics(keywords, max_length=120, temperature=1.0):
    """Generate lyrics using GPT-2 conditioned on the provided keywords."""
    # Create prompt
    prompt = "Write a song about " + ", ".join(keywords) + ":\n"

    # Load text-generation pipeline
    generator = pipeline("text-generation", model="gpt2")

    # Set a seed for reproducibility
    seed = random.randint(0, 2**32 - 1)
    set_seed(seed)

    # Generate text
    generated = generator(
        prompt,
        max_length=len(prompt.split()) + max_length,
        num_return_sequences=1,
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        eos_token_id=None,
    )

    # Extract the generated portion beyond the prompt
    text = generated[0]["generated_text"]
    lyrics = text[len(prompt):].strip()
    return lyrics


def generate_melody(filename="melody.mid", num_notes=16, tempo=120):
    """Generate a simple melody saved as a MIDI file."""
    track = 0
    channel = 0
    time = 0
    duration = 1
    volume = 100

    midi = MIDIFile(1)
    midi.addTempo(track, time, tempo)

    # Define a C major scale (MIDI note numbers)
    scale = [60, 62, 64, 65, 67, 69, 71, 72]

    for i in range(num_notes):
        pitch = random.choice(scale)
        midi.addNote(track, channel, pitch, time + i, duration, volume)

    with open(filename, "wb") as out_file:
        midi.writeFile(out_file)


def main():
    parser = argparse.ArgumentParser(description="Generate a song from keywords.")
    parser.add_argument(
        "keywords",
        nargs="+",
        help="Keywords to inspire the song (e.g. love, adventure, summer).",
    )
    parser.add_argument(
        "--lyrics-length",
        type=int,
        default=120,
        help="Number of tokens to generate for lyrics.",
    )
    parser.add_argument(
        "--melody-notes",
        type=int,
        default=16,
        help="Number of notes in the generated melody.",
    )
    parser.add_argument(
        "--output-melody",
        default="melody.mid",
        help="Filename for the generated MIDI melody.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature parameter controlling creativity of lyrics (higher = more creative).",
    )
    args = parser.parse_args()

    lyrics = generate_lyrics(args.keywords, args.lyrics_length, args.temperature)
    generate_melody(args.output_melody, args.melody_notes)

    print("Generated Lyrics:\n")
    print(lyrics)
    print(f"\nMelody saved to {args.output_melody}")


if __name__ == "__main__":
    main()

import gradio as gr
import requests
import json

API_URL = "http://localhost:8000"

def analyze_text(text, analysis_type):
    """
    Send text to your FastAPI backend based on analysis type
    """
    if analysis_type == "Complete Analysis (All features)":
        # Call all three endpoints
        analyze_response = requests.post(f"{API_URL}/analyze", json={"text": text})
        caption_response = requests.post(f"{API_URL}/analyze-caption", json={"text": text})
        tags_response = requests.post(f"{API_URL}/generate-tags", json={"text": text})
        
        if analyze_response.status_code == 200:
            analyze_data = analyze_response.json()
            caption_data = caption_response.json() if caption_response.status_code == 200 else {"caption": "Error", "summary": "Error", "sentiment": "Error"}
            tags_data = tags_response.json() if tags_response.status_code == 200 else {"tags": []}
            
            result = f"""
            📝 **SUMMARY**
            {analyze_data.get('summary', 'N/A')}
            
            💡 **KEY INSIGHTS**
            {chr(10).join(['• ' + insight for insight in analyze_data.get('insights', [])])}
            
            🎭 **SENTIMENT**
            {analyze_data.get('sentiment', 'N/A').upper()}
            
            ✨ **AI-GENERATED CAPTION**
            {caption_data.get('caption', 'N/A')}
            
            🏷️ **SUGGESTED TAGS**
            {' '.join(['#' + tag for tag in tags_data.get('tags', [])])}
            """
            return result
        else:
            return f"Error: {analyze_response.status_code}"
    
    elif analysis_type == "Summary + Sentiment":
        response = requests.post(f"{API_URL}/analyze", json={"text": text})
        if response.status_code == 200:
            data = response.json()
            return f"📝 **Summary:** {data['summary']}\n\n🎭 **Sentiment:** {data['sentiment'].upper()}\n\n💡 **Insights:**\n" + '\n'.join(['• ' + i for i in data['insights']])
        return f"Error: {response.status_code}"
    
    elif analysis_type == "Caption Generator":
        response = requests.post(f"{API_URL}/analyze-caption", json={"text": text})
        if response.status_code == 200:
            data = response.json()
            return f"✨ **Caption:** {data['caption']}\n\n📝 **Summary:** {data['summary']}\n\n🎭 **Sentiment:** {data['sentiment'].upper()}"
        return f"Error: {response.status_code}"
    
    else:  # Tags only
        response = requests.post(f"{API_URL}/generate-tags", json={"text": text})
        if response.status_code == 200:
            data = response.json()
            tags = ' '.join(['#' + tag for tag in data['tags']])
            return f"🏷️ **Suggested Tags:**\n{tags}"
        return f"Error: {response.status_code}"

        # Create the Gradio interface
with gr.Blocks(title="AI Text Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 AI Text Analyzer
    
    ### Powered by OpenAI + LangChain
    
    Paste any text below and get AI-powered insights, summaries, captions, and hashtags!
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="📄 Your Text",
                placeholder="Paste your article, video description, or any text here...\n\nExample: LeBron James scored 40 points and led the Lakers to an incredible comeback victory in the final minutes.",
                lines=10
            )
            
            analysis_type = gr.Radio(
                choices=["Complete Analysis (All features)", "Summary + Sentiment", "Caption Generator", "Tags Only"],
                label="🔧 Analysis Type",
                value="Complete Analysis (All features)"
            )
            
            analyze_btn = gr.Button("🚀 Analyze Text", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            output = gr.Markdown(label="📊 Analysis Results")
    
    # Examples
    gr.Markdown("### 📝 Try these examples:")
    gr.Examples(
        examples=[
            ["LeBron James dominated with 45 points and the game-winning shot in the final seconds of Game 7."],
            ["The new iPhone features an amazing camera and incredible battery life. I'm so excited to buy it!"],
            ["The company reported disappointing earnings, leading to a 15% drop in stock price. Investors are concerned."],
            ["A beautiful sunset at the beach with waves gently crashing on the shore. Perfect peaceful evening."]
        ],
        inputs=text_input,
        label="Click any example to test"
    )
    
    # Connect the button to the function
    analyze_btn.click(
        fn=analyze_text,
        inputs=[text_input, analysis_type],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### 🚀 Powered by:
    - **FastAPI** backend with LangChain
    - **OpenAI GPT** for text analysis
    - **Gradio** for this beautiful interface
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
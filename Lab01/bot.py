import os
import telebot
from dotenv import load_dotenv
import re
import traceback
import data_manager
import nlp_core
import visualizer
import classifier
from nltk.tokenize import sent_tokenize

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise ValueError("Missing token! Make sure .env contains TELEGRAM_BOT_TOKEN.")

bot = telebot.TeleBot(TOKEN)

CLASS_ALIASES = {
    "pozytywny": "pozytywny",
    "neutralny": "neutralny",
    "negatywny": "negatywny",
}

def extract_args(text, expected_args_count):
    # Find all sequences between double quotes.
    matches = re.findall(r'"([^"]*)"', text)
    if len(matches) == expected_args_count:
        return matches
    return None

def normalize_and_validate_class(raw_class):
    normalized = raw_class.strip().lower()
    return CLASS_ALIASES.get(normalized)

def log_exception(context, error):
    print(f"[ERROR] {context}: {type(error).__name__}: {error}")
    print(traceback.format_exc())

def get_safe_error_message(error):
    if isinstance(error, LookupError):
        return "NLP resources are missing on the server. Please contact the administrator."
    return "An internal processing error occurred. Please try again later."

def handle_exception(message, context, error):
    log_exception(context, error)
    bot.reply_to(message, get_safe_error_message(error))

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = (
        "Hi! I am your NLP bot (Lab 1).\n\n"
        "Available commands:\n"
        "/task <task_name> \"text\" \"class\"\n"
        "/full_pipeline \"text\" \"class\"\n"
        "/classifier \"text\"\n"
        "/stats\n\n"
        "Tasks for /task:\n"
        "tokenize, remove_stopwords, lemmatize, stemming, stats, n-grams, plot_histogram, plot_wordcloud, plot_barchart"
    )
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['task'])
def handle_task(message):
    try:
        # Parse command according to docs: /task <task_name> "text" "class"
        parts = message.text.split(maxsplit=2)
        if len(parts) < 3:
            bot.reply_to(message, "Invalid format! Use: /task <task_name> \"text\" \"class\"")
            return
            
        task_name = parts[1]
        args_text = parts[2]
        
        extracted = extract_args(args_text, 2)
        if not extracted:
            bot.reply_to(message, "Could not parse arguments. Put text and class in quotes, e.g. /task tokenize \"Sample text\" \"pozytywny\".")
            return
             
        user_text, text_class = extracted[0], extracted[1]
        
        if not user_text.strip():
            bot.reply_to(message, "Provided message text is empty!")
            return

        normalized_class = normalize_and_validate_class(text_class)
        if not normalized_class:
            bot.reply_to(message, "Invalid class. Allowed classes: pozytywny, neutralny, negatywny.")
            return

        # 1. Execute NLP or visualization task.
        response_msg = f"Running task: {task_name}\n"
        
        visualizations = ["plot_histogram", "plot_wordcloud", "plot_barchart"]
        nlp_tasks = ["tokenize", "remove_stopwords", "lemmatize", "stemming", "stats", "n-grams"]
        
        plot_path = None
        
        if task_name in nlp_tasks:
            result = nlp_core.run_task(task_name, user_text)
            response_msg += f"Result:\n{result}"
            
        elif task_name in visualizations:
            tokens = nlp_core.tokenize_text(user_text)
            clean_tokens = [t for t in tokens if t not in '.,!?;:()[]"\'']
            
            if task_name == "plot_histogram":
                plot_path = visualizer.plot_token_length_histogram(clean_tokens)
                response_msg += "Histogram was generated." if plot_path else "Histogram could not be generated for this input."
            elif task_name == "plot_wordcloud":
                plot_path = visualizer.plot_wordcloud(user_text)
                response_msg += "Word cloud was generated." if plot_path else "Word cloud could not be generated for this input."
            elif task_name == "plot_barchart":
                plot_path = visualizer.plot_most_common_words(clean_tokens)
                response_msg += "Bar chart was generated." if plot_path else "Bar chart could not be generated for this input."
        else:
             bot.reply_to(message, f"Unknown task: {task_name}. Use /help to see available tasks.")
             return
        
        # 2. Save labeled example to JSON.
        data_manager.save_record(user_text, normalized_class)
        
        # 3. Return response (text and optional image file).
        bot.reply_to(message, response_msg)
        
        if plot_path and os.path.exists(plot_path):
            with open(plot_path, 'rb') as photo:
                bot.send_photo(message.chat.id, photo)

    except Exception as e:
        handle_exception(message, "task", e)

@bot.message_handler(commands=['full_pipeline'])
def handle_full_pipeline(message):
    try:
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            bot.reply_to(message, "Invalid format! Use: /full_pipeline \"text\" \"class\"")
            return
            
        args_text = parts[1]
        extracted = extract_args(args_text, 2)
        if not extracted:
            bot.reply_to(message, "Invalid quote format. Example: /full_pipeline \"Sample text\" \"pozytywny\"")
            return
             
        user_text, text_class = extracted[0], extracted[1]

        if not user_text.strip():
            bot.reply_to(message, "Provided message text is empty!")
            return

        normalized_class = normalize_and_validate_class(text_class)
        if not normalized_class:
            bot.reply_to(message, "Invalid class. Allowed classes: pozytywny, neutralny, negatywny.")
            return
        
        # Save to JSON with sentence-level split.
        sentences = sent_tokenize(user_text, language="polish")
        for s in sentences:
            data_manager.save_record(s, normalized_class)

        # 1. Cleaning and basic preprocessing.
        tokens = nlp_core.tokenize_text(user_text)
        clean_tokens = [t for t in tokens if t not in '.,!?;:()[]"\'']
        
        # Pipeline computations.
        no_stop = nlp_core.remove_stopwords(clean_tokens)
        lemmas = nlp_core.lemmatize(clean_tokens)
        stems = nlp_core.stemming(clean_tokens)
        bow_feat, bow_vec = nlp_core.bag_of_words(user_text)
        tfidf_feat, tfidf_vec = nlp_core.tf_idf(user_text)
        stats = nlp_core.get_stats(clean_tokens)

        response = (
            f"--- FULL PIPELINE ---\n"
            f"1. Saved entries (sentences: {len(sentences)})\n"
            f"2. Tokens: {tokens[:10]}...\n"
            f"3. Without stopwords (Top10): {no_stop[:10]}...\n"
            f"4. Lemmatization (Top10): {lemmas[:10]}...\n"
            f"5. Stemming (Top10): {stems[:10]}...\n"
            f"6. Bag of words: shape {bow_vec.shape if len(bow_vec)>0 else 'None'}\n"
            f"7. TF-IDF: shape {tfidf_vec.shape if len(tfidf_vec)>0 else 'None'}\n"
            f"8. Statistics: {stats}"
        )
        bot.reply_to(message, response)

        # Generate 3 required charts.
        p1 = visualizer.plot_most_common_words(clean_tokens)
        p2 = visualizer.plot_token_length_histogram(clean_tokens)
        p3 = visualizer.plot_wordcloud(user_text)

        for p in [p1, p2, p3]:
            if p and os.path.exists(p):
                with open(p, 'rb') as photo:
                    bot.send_photo(message.chat.id, photo)

    except Exception as e:
        handle_exception(message, "full_pipeline", e)

@bot.message_handler(commands=['classifier'])
def handle_classifier(message):
    try:
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            bot.reply_to(message, "Use: /classifier \"text_to_classify\"")
            return
        
        extracted = extract_args(parts[1], 1)
        if not extracted:
            user_text = parts[1].replace('"', '')  # Fallback
        else:
            user_text = extracted[0]

        if not user_text.strip():
            bot.reply_to(message, "Provided message text is empty!")
            return

        prediction = classifier.train_and_predict(user_text)
        bot.reply_to(message, f"Classifier prediction:\n➡️ {prediction}")

    except Exception as e:
        handle_exception(message, "classifier", e)

@bot.message_handler(commands=['stats'])
def handle_stats(message):
    try:
        records = data_manager.load_records()
        if not records:
            bot.reply_to(message, "No data in dataset. Use /task or /full_pipeline first.")
            return

        all_text = " ".join([r["text"] for r in records])
        labels = [r["class"] for r in records]
        
        class_counts = {k: labels.count(k) for k in set(labels)}
        class_stats_str = "\n".join([f"- {k}: {v}" for k, v in class_counts.items()])

        tokens = nlp_core.tokenize_text(all_text)
        clean_tokens = [t.lower() for t in tokens if t not in '.,!?;:()[]"\'-']
        unique_tokens = set(clean_tokens)
        
        bigrams = list(set(nlp_core.get_ngrams(clean_tokens, 2)))
        trigrams = list(set(nlp_core.get_ngrams(clean_tokens, 3)))

        response = (
            f"**DATASET STATISTICS**\n\n"
            f"Class distribution:\n{class_stats_str}\n\n"
            f"Unique tokens (Top 10 of {len(unique_tokens)}): {list(unique_tokens)[:10]}...\n\n"
            f"Unique 2-grams (Top 5): {bigrams[:5]}...\n"
            f"Unique 3-grams (Top 5): {trigrams[:5]}..."
        )
        bot.reply_to(message, response)

        p1 = visualizer.plot_most_common_words(clean_tokens)
        p2 = visualizer.plot_token_length_histogram(clean_tokens)
        p3 = visualizer.plot_wordcloud(all_text)

        for p in [p1, p2, p3]:
            if p and os.path.exists(p):
                with open(p, 'rb') as photo:
                    bot.send_photo(message.chat.id, photo)

    except Exception as e:
        handle_exception(message, "stats", e)

if __name__ == '__main__':
    print("Bot is starting...")
    # Polling keeps the bot listening for new messages.
    bot.infinity_polling()

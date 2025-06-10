import requests
import re
import time
import random
import csv
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Ethical and configuration settings
MAX_SENTENCES = 1000
ANONYMIZE = True  # Remove usernames/IDs
REQUEST_DELAY = 2  # Seconds between requests
VERBOSE = True  # Enable verbose output

# Expanded hate speech keywords by category
HATE_KEYWORDS = {
    'sexism': ['slut', 'whore', 'bitch', 'cunt', 'feminazi', 'make me a sandwich', 'women belong', 'tits', 'rape', 'femicunt', 'roastie', 'thot', 'gold digger', 'attention whore', 'basic bitch'],
    'racism': ['nigger', 'chink', 'spic', 'kike', 'coon', 'white power', 'monkey', 'beaner', 'jungle bunny', 'mud people', 'wetback', 'gook', 'raghead', 'porch monkey', 'cotton picker'],
    'nationalism': ['deport', 'send back', 'invaders', 'immigrant filth', 'build the wall', 'foreign scum', 'go back to', 'pure blood', 'white genocide', 'great replacement', 'anchor baby', 'illegals'],
    'antisemitism': ['kike', 'jew rat', 'holohoax', 'oven dodger', 'blood libel', 'globalist jew', 'jewish conspiracy', 'shekel', 'goyim', 'nose tribe', 'merchant', 'six gorillion'],
    'islamophobia': ['towelhead', 'sand nigger', 'bomb thrower', 'muzzie', 'terrorist', 'allah snackbar', 'islam is cancer', 'goat fucker', 'kebab', 'mohammed was a pedophile'],
    'homophobia': ['faggot', 'fag', 'dyke', 'tranny', 'homo', 'queer', 'aids spreader', 'buttfucker', 'carpet muncher', 'pillow biter', 'rainbow mafia', 'groomer'],
    'transphobia': ['tranny', 'trap', 'transformer', 'mentally ill', 'delusional', 'chopped off', 'mutilated', 'fake woman', 'man in dress', 'trojan horse'],
    'ableism': ['retard', 'retarded', 'sperg', 'autist', 'downs', 'mongoloid', 'cripple', 'gimp', 'psycho', 'mental case', 'brain dead'],
    'classism': ['trailer trash', 'white trash', 'welfare queen', 'food stamp', 'ghetto rat', 'hood rat', 'peasant', 'poor scum', 'bottom feeder'],
    'general': ['die ', 'kill ', 'exterminate', 'scum', 'trash', 'vermin', 'subhuman', 'worthless', 'degenerate', 'parasite', 'cancer', 'plague', 'disease'],
    'class': ['trailer trash', 'white trash', 'welfare queen', 'food stamp', 'ghetto rat', 'hood rat', 'peasant', 'poor scum', 'bottom feeder']
}

# Bad words/profanity (for non-hate cases)
BAD_WORDS = [
    'fuck', 'shit', 'damn', 'hell', 'ass', 'piss', 'cock', 'dick', 'pussy', 'balls', 
    'bullshit', 'motherfucker', 'asshole', 'bastard', 'douchebag', 'jackass', 
    'dickhead', 'shithead', 'fuckface', 'dumbass', 'dipshit', 'prick', 'turd',
    'crap', 'goddamn', 'jesus christ', 'holy shit', 'what the fuck', 'son of a bitch'
]

# Combine all hate keywords for filtering
ALL_HATE_KEYWORDS = set()
for category in HATE_KEYWORDS.values():
    ALL_HATE_KEYWORDS.update(category)

# Initialize data collection
collected_sentences = set()
collected_data = []

def verbose_print(message):
    """Print message if verbose mode is enabled"""
    if VERBOSE:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

def contains_hate(text):
    """Check if text contains hate keywords (case-insensitive)"""
    text_lower = text.lower()
    return any(re.search(rf'\b{re.escape(keyword)}\b', text_lower) 
               for keyword in ALL_HATE_KEYWORDS)

def contains_profanity(text):
    """Check if text contains profanity/bad words (case-insensitive)"""
    text_lower = text.lower()
    return any(re.search(rf'\b{re.escape(word)}\b', text_lower) 
               for word in BAD_WORDS)

def clean_html_entities(text):
    """Clean HTML entities and special symbols"""
    # Convert common HTML entities
    text = text.replace('&gt;', '>')
    text = text.replace('&lt;', '<')
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    text = text.replace('&#039;', "'")
    text = text.replace('&nbsp;', ' ')
    
    # Remove >>numbers patterns (4chan post references)
    text = re.sub(r'&gt;&gt;\d+', '[ID]', text)
    text = re.sub(r'>>\d+', '[ID]', text)
    
    return text

def extract_sentences(text, collect_profanity=True):
    """Split text into sentences and filter interesting ones"""
    if not text:
        return []
    
    # Clean HTML entities first
    text = clean_html_entities(text)
    
    # Improved sentence splitting
    sentences = []
    
    # Split by sentence endings
    raw_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+', text)
    
    # Also split by line breaks for shorter posts
    for sent in raw_sentences:
        if '\n' in sent:
            sentences.extend([s.strip() for s in sent.split('\n')])
        else:
            sentences.append(sent.strip())
    
    results = []
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 8:  # Lowered minimum length
            continue
            
        # Check for hate speech
        if contains_hate(sent):
            results.append((sent, 'hate'))
        # Check for profanity (non-hate)
        elif collect_profanity and contains_profanity(sent):
            results.append((sent, 'profanity'))
    
    return results

def anonymize_text(text):
    """Remove usernames and IDs from text"""
    text = re.sub(r'@\w+', '[USER]', text)  # Mentions
    text = re.sub(r'>>\d+', '[ID]', text)    # 4chan references
    text = re.sub(r'#\w+', '[HASHTAG]', text)  # Hashtags
    text = re.sub(r'https?://\S+', '[URL]', text)  # URLs
    text = re.sub(r'&gt;&gt;\d+', '[ID]', text)  # HTML entity references
    return text

def is_hateful_thread(thread_title, thread_comment=""):
    """Check if thread title or comment suggests hateful content"""
    hateful_indicators = [
        'jews', 'nigger', 'muslim', 'immigrant', 'woman', 'feminist', 'tranny', 'trans', 'fag',
        'hate', 'kill', 'genocide', 'white', 'black', 'arab', 'mexican', 'asian', 'gay', 'lesbian',
        'race', 'religion', 'gender', 'libtard', 'cuck', 'soy', 'degener', 'subhuman', 'based',
        'redpill', 'blackpill', 'incel', 'mgtow', 'alt-right', 'nationalism', 'supremacy'
    ]
    text = (thread_title + " " + thread_comment).lower()
    return any(indicator in text for indicator in hateful_indicators)

# ======================
# 4CHAN SCRAPER (Multiple boards)
# ======================
def scrape_4chan():
    verbose_print("Starting 4chan scraping...")
    
    # Target boards known for controversial content
    boards = ['pol', 'b', 'r9k']
    
    for board in boards:
        verbose_print(f"Scraping 4chan /{board}/ board...")
        if len(collected_data) >= MAX_SENTENCES:
            break
            
        try:
            # Get active threads
            threads_url = f"https://a.4cdn.org/{board}/threads.json"
            response = requests.get(threads_url, timeout=10)
            threads = response.json()[0]['threads']
            
            # Sort threads by reply count and randomize top threads for better hate content
            threads = sorted(threads, key=lambda x: x.get('replies', 0), reverse=True)
            random.shuffle(threads[:30])  # Increased to top 30 threads
            
            verbose_print(f"Found {len(threads)} threads in /{board}/, checking top active ones...")
            
            thread_count = 0
            for thread in threads[:30]:  # Check more threads
                if len(collected_data) >= MAX_SENTENCES:
                    break
                    
                thread_count += 1
                thread_url = f"https://a.4cdn.org/{board}/thread/{thread['no']}.json"
                
                try:
                    thread_data = requests.get(thread_url, timeout=10).json()
                    
                    # Check if thread seems hateful based on OP post
                    op_post = thread_data['posts'][0]
                    thread_title = op_post.get('sub', '')
                    thread_comment = op_post.get('com', '')
                    
                    # More lenient filtering - check more threads
                    if not is_hateful_thread(thread_title, thread_comment) and random.random() > 0.5:
                        continue  # Skip 50% of non-hateful threads (was 70%)
                    
                    verbose_print(f"  Processing thread {thread_count}/30 in /{board}/ (Thread #{thread['no']})")
                    
                    post_count = 0
                    hate_sentences_found = 0
                    profanity_sentences_found = 0
                    
                    for post in thread_data['posts']:
                        post_count += 1
                        
                        # Get post text from comment
                        raw_text = post.get('com', '')
                        if not raw_text:
                            continue
                        
                        # Clean HTML tags but preserve structure
                        text = re.sub(r'<br\s*/?>', '\n', raw_text)  # Convert <br> to newlines
                        text = re.sub(r'<[^<]+?>', '', text)  # Remove other HTML
                        text = clean_html_entities(text)  # Clean HTML entities
                        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
                        
                        if not text or len(text) < 5:
                            continue
                            
                        if ANONYMIZE:
                            text = anonymize_text(text)
                        
                        # Process sentences
                        sentence_results = extract_sentences(text, collect_profanity=True)
                        for sentence, category in sentence_results:
                            if len(collected_data) >= MAX_SENTENCES:
                                break
                            if sentence not in collected_sentences:
                                collected_sentences.add(sentence)
                                label = 'hate' if category == 'hate' else 'non-hate'
                                collected_data.append([sentence, f'4chan-{board}', label])
                                
                                if category == 'hate':
                                    hate_sentences_found += 1
                                else:
                                    profanity_sentences_found += 1
                    
                    if hate_sentences_found > 0 or profanity_sentences_found > 0:
                        verbose_print(f"    Found {hate_sentences_found} hate + {profanity_sentences_found} profanity sentences from {post_count} posts")
                    
                    time.sleep(REQUEST_DELAY + random.uniform(0, 1))
                    
                except Exception as e:
                    verbose_print(f"    Error processing thread {thread['no']}: {str(e)}")
                    continue
                    
        except Exception as e:
            verbose_print(f"Error accessing /{board}/ board: {str(e)}")
            continue
    
    verbose_print(f"4chan scraping complete. Total collected: {len(collected_data)} sentences")

# ===============
# ENHANCED KIWIFARMS SCRAPER - MAXIMUM HATE EXTRACTION
# ===============
def scrape_kiwifarms():
    verbose_print("Starting KiwiFarms scraping (targeting high-hate content)...")
    
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu') 
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    try:
        # Test basic access
        verbose_print("Testing basic KiwiFarms access...")
        driver.get("https://kiwifarms.st/")
        time.sleep(5)
        
        verbose_print(f"  Successfully connected to KiwiFarms")
        
        # MASSIVE EXPANDED HATE SEARCH TERMS - KiwiFarms specific
        hate_search_categories = {
            'transgender_hate': [
                'transgender', 'tranny', 'troon', 'trans woman', 'trans man', 
                'hormone therapy', 'gender dysphoria', 'transition', 'top surgery',
                'bottom surgery', 'neo vagina', 'dilate', 'passing', 'clockable'
            ],
            'racial_slurs': [
                'nigger', 'negro', 'coon', 'monkey', 'ape', 'dindu', 'jogger',
                'basketball american', 'urban youth', 'thugs', 'chimping', 'gibs'
            ],
            'antisemitic': [
                'jew', 'kike', 'shekel', 'goyim', 'rabbi', 'nose', 'merchant',
                'protocols', 'holohoax', 'oven', 'lampshade', 'six million'
            ],
            'homophobic': [
                'faggot', 'fag', 'homo', 'queer', 'aids', 'buttfucker', 
                'dyke', 'carpet muncher', 'rainbow mafia', 'pride month'
            ],
            'misogynistic': [
                'feminist', 'feminism', 'roastie', 'thot', 'whore', 'slut',
                'femoid', 'foid', 'hole', 'used goods', 'body count'
            ],
            'ableist': [
                'retard', 'retarded', 'sperg', 'autistic', 'downs', 'mongoloid',
                'cripple', 'wheelchair', 'vegetable', 'braindead'
            ],
            'religious_hate': [
                'muslim', 'islam', 'mohammed', 'allah', 'hijab', 'burqa',
                'terrorist', 'bomber', 'goat fucker', 'sand nigger'
            ],
            'political_hate': [
                'immigrant', 'illegal', 'deportation', 'build wall', 'invasion',
                'replacement', 'white genocide', 'diversity', 'multiculturalism'
            ],
            'violence_advocacy': [
                'kill', 'murder', 'lynch', 'hang', 'shoot', 'stab', 'rape',
                'torture', 'genocide', 'exterminate', 'gas chamber'
            ],
            'general_dehumanizing': [
                'subhuman', 'degenerate', 'parasite', 'vermin', 'cancer',
                'disease', 'plague', 'scum', 'trash', 'worthless'
            ],
            'kiwifarms_specific': [
                'lolcow', 'chris chan', 'cwcville', 'sonichu', 'dimensional merge',
                'exceptional individual', 'a-log', 'weening', 'cowtipping'
            ],
            'grooming_predator': [
                'groomer', 'predator', 'pedophile', 'child abuse', 'minor attracted',
                'cp', 'cheese pizza', 'loli', 'shota', 'age of consent'
            ],
            'class': ['trailer trash', 'white trash', 'welfare queen', 'food stamp', 'ghetto rat', 'hood rat', 'peasant', 'poor scum', 'bottom feeder']
        }
        
        # Flatten all search terms and randomize
        all_search_terms = []
        for category, terms in hate_search_categories.items():
            for term in terms:
                all_search_terms.append((term, category))
        
        random.shuffle(all_search_terms)
        
        verbose_print(f"📊 Generated {len(all_search_terms)} targeted hate search terms")
        verbose_print(f"🎯 Will process up to 15 searches for maximum hate extraction")
        
        # Process significantly more searches
        searches_to_run = min(15, len(all_search_terms))  # Increased from 4 to 15
        
        for search_idx in range(searches_to_run):
            if len(collected_data) >= MAX_SENTENCES:
                verbose_print(f"🎯 Reached target of {MAX_SENTENCES} sentences")
                break
                
            search_term, category = all_search_terms[search_idx]
            verbose_print(f"\n  🎯 Search {search_idx+1}/{searches_to_run}: '{search_term}' ({category})")
            
            try:
                search_url = f"https://kiwifarms.st/search/search/?keywords={search_term}&order=date"
                verbose_print(f"    Navigating to search...")
                driver.get(search_url)
                time.sleep(4)  # Slightly longer wait
                
                if "search" not in driver.current_url:
                    verbose_print(f"    ❌ Search failed for '{search_term}'")
                    continue
                
                # Get search result count
                result_elements = driver.find_elements(By.CSS_SELECTOR, ".contentRow-title a")
                total_results = len(result_elements)
                verbose_print(f"    Found {total_results} search results")
                
                if total_results == 0:
                    verbose_print(f"    ❌ No results for '{search_term}'")
                    continue
                
                # Process MORE results per search (increased from 3 to 5)
                results_to_process = min(5, total_results)
                
                search_hate_total = 0
                search_profanity_total = 0
                
                for result_idx in range(results_to_process):
                    if len(collected_data) >= MAX_SENTENCES:
                        break
                        
                    try:
                        verbose_print(f"      Processing result {result_idx+1}/{results_to_process}...")
                        
                        # Go back to search results
                        driver.get(search_url)
                        time.sleep(2)
                        
                        # Re-find results
                        fresh_results = driver.find_elements(By.CSS_SELECTOR, ".contentRow-title a")
                        if result_idx >= len(fresh_results):
                            verbose_print(f"        ❌ Result {result_idx+1} unavailable")
                            continue
                        
                        result_link = fresh_results[result_idx]
                        result_url = result_link.get_attribute('href')
                        result_title = result_link.text.strip()
                        
                        verbose_print(f"        Title: {result_title[:60]}...")
                        
                        if not result_url or 'threads/' not in result_url:
                            verbose_print(f"        ❌ Not a thread URL")
                            continue
                        
                        # Navigate to thread
                        driver.get(result_url)
                        time.sleep(3)
                        
                        # Extract MORE posts per thread (increased from 15 to 30)
                        posts = driver.find_elements(By.CSS_SELECTOR, ".message-body .bbWrapper, .message-userContent")
                        posts_to_check = min(30, len(posts))  # Process up to 30 posts
                        verbose_print(f"        Processing {posts_to_check} posts...")
                        
                        result_hate_found = 0
                        result_profanity_found = 0
                        
                        for post_idx, post in enumerate(posts[:posts_to_check]):
                            try:
                                text = post.text.strip()
                                if not text or len(text) < 8:  # Lower threshold
                                    continue
                                
                                text = clean_html_entities(text)
                                if ANONYMIZE:
                                    text = anonymize_text(text)
                                
                                sentence_results = extract_sentences(text, collect_profanity=True)
                                for sentence, sentence_category in sentence_results:
                                    if len(collected_data) >= MAX_SENTENCES:
                                        break
                                    if sentence not in collected_sentences:
                                        collected_sentences.add(sentence)
                                        label = 'hate' if sentence_category == 'hate' else 'non-hate'
                                        collected_data.append([sentence, 'KiwiFarms', label])
                                        
                                        if sentence_category == 'hate':
                                            result_hate_found += 1
                                            verbose_print(f"          ✅ HATE [{category}]: {sentence[:50]}...")
                                        else:
                                            result_profanity_found += 1
                            except Exception as e:
                                continue
                        
                        search_hate_total += result_hate_found
                        search_profanity_total += result_profanity_found
                        
                        if result_hate_found > 0 or result_profanity_found > 0:
                            verbose_print(f"        📊 Result: {result_hate_found} hate + {result_profanity_found} profanity")
                        else:
                            verbose_print(f"        📊 No relevant content found")
                        
                    except Exception as e:
                        verbose_print(f"      ❌ Error processing result {result_idx+1}: {str(e)}")
                        continue
                
                verbose_print(f"    🎯 Search '{search_term}' total: {search_hate_total} hate + {search_profanity_total} profanity")
                
                # Add small delay between searches
                time.sleep(1)
                        
            except Exception as e:
                verbose_print(f"  ❌ Error in search for '{search_term}': {str(e)}")
                continue
        
        # BONUS: Try some multi-word hate phrases for maximum extraction
        verbose_print(f"\n🔥 BONUS ROUND: Multi-word hate phrases")
        
        hate_phrases = [
            'trans women are men', 'biological male', 'mentally ill tranny',
            'grooming children', 'sexual predator', 'degenerate faggot',
            'stupid nigger', 'jewish conspiracy', 'white genocide',
            'kill all', 'rape victim', 'subhuman scum'
        ]
        
        for phrase_idx, phrase in enumerate(hate_phrases[:5]):  # Top 5 phrases
            if len(collected_data) >= MAX_SENTENCES:
                break
                
            verbose_print(f"  🔥 Bonus search {phrase_idx+1}/5: '{phrase}'")
            
            try:
                search_url = f"https://kiwifarms.st/search/search/?keywords={phrase}&order=relevance"
                driver.get(search_url)
                time.sleep(3)
                
                results = driver.find_elements(By.CSS_SELECTOR, ".contentRow-title a")[:2]  # Top 2 most relevant
                
                for result in results:
                    try:
                        driver.get(result.get_attribute('href'))
                        time.sleep(2)
                        
                        posts = driver.find_elements(By.CSS_SELECTOR, ".message-body .bbWrapper")[:20]
                        
                        for post in posts:
                            text = clean_html_entities(post.text.strip())
                            if ANONYMIZE:
                                text = anonymize_text(text)
                            
                            sentence_results = extract_sentences(text, collect_profanity=True)
                            for sentence, category in sentence_results:
                                if sentence not in collected_sentences:
                                    collected_sentences.add(sentence)
                                    label = 'hate' if category == 'hate' else 'non-hate'
                                    collected_data.append([sentence, 'KiwiFarms', label])
                                    
                                    if category == 'hate':
                                        verbose_print(f"    🔥 BONUS HATE: {sentence[:50]}...")
                    except:
                        continue
                        
            except Exception as e:
                verbose_print(f"    ❌ Bonus search error: {str(e)}")
                continue
                
    except Exception as e:
        verbose_print(f"❌ KiwiFarms scraping error: {str(e)}")
    finally:
        driver.quit()
    
    kiwi_count = len([item for item in collected_data if item[1] == 'KiwiFarms'])
    kiwi_hate_count = len([item for item in collected_data if item[1] == 'KiwiFarms' and item[2] == 'hate'])
    verbose_print(f"\n=== KIWIFARMS SCRAPING COMPLETE ===")
    verbose_print(f"📊 Total sentences: {kiwi_count}")
    verbose_print(f"🎯 Hate sentences: {kiwi_hate_count}")
    verbose_print(f"💬 Non-hate sentences: {kiwi_count - kiwi_hate_count}")
    verbose_print(f"📈 Hate rate: {(kiwi_hate_count/kiwi_count*100):.1f}%" if kiwi_count > 0 else "�� Hate rate: 0%")

# =================
# MAIN EXECUTION - FIXED ORDER
# =================
if __name__ == "__main__":
    verbose_print("Starting hate speech data collection...")
    verbose_print(f"Target: {MAX_SENTENCES} sentences")
    verbose_print(f"Total hate keywords: {len(ALL_HATE_KEYWORDS)}")
    verbose_print(f"Total profanity words: {len(BAD_WORDS)}")
    
    # Run 4chan scraper FIRST
    verbose_print("\n=== PHASE 1: 4CHAN SCRAPING ===")
    scrape_4chan()
    
    # Then run KiwiFarms if needed
    if len(collected_data) < MAX_SENTENCES:
        verbose_print(f"\n=== PHASE 2: KIWIFARMS SCRAPING ===")
        verbose_print(f"Need {MAX_SENTENCES - len(collected_data)} more sentences, trying KiwiFarms...")
        scrape_kiwifarms()

    # Save results
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, 'parsed_data.csv')
    with open(output_file, 'a+', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'source', 'label'])
        writer.writerows(collected_data)
    
    # Summary statistics
    sources = {}
    labels = {}
    for item in collected_data:
        source = item[1]
        label = item[2]
        sources[source] = sources.get(source, 0) + 1
        labels[label] = labels.get(label, 0) + 1
    
    verbose_print(f"\n=== COLLECTION COMPLETE ===")
    verbose_print(f"Total sentences collected: {len(collected_data)}")
    verbose_print("Source breakdown:")
    for source, count in sources.items():
        verbose_print(f"  {source}: {count} sentences")
    verbose_print("Label breakdown:")
    for label, count in labels.items():
        verbose_print(f"  {label}: {count} sentences")
    verbose_print(f"Saved to: {output_file}")
    verbose_print(f"Unique sentences: {len(collected_sentences)}")
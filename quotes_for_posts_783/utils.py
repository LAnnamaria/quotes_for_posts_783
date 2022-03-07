import quotes_for_posts_783.data.quotesdata as qd

def image_cap_to_quotes(quotes,image_caption):
    
    quotes.iloc[-1] = [image_caption, 'image','image',qd.remove_punctuations(image_caption),'1']
    return quotes
    
if __name__ == '__main__':
    image_caption = 'I ??am standing in from of the Eiffel tower with a buoqet of roses.'
    quotes = qd.get_quotes_data()
    quotes = qd.clean_data(quotes)
    quotes = image_cap_to_quotes(quotes,image_caption)
    print(quotes.list_tags.tail(3))
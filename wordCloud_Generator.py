from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordCloud():
    # yelp
    # texts = open("reviews.txt","r").readlines()
    # enron
    texts = open("content.txt","r").readlines()

    # Generate a word cloud image
    text_join = "".join(texts)
    print(type(text_join))

    # generate wordcloud with default parameters
    wordcloud = WordCloud().generate(text_join)
    # Display the generated image
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # generate wordcloud with lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text_join)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    generate_wordCloud()
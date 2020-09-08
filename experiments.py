from util_funcs import *
from transformers import pipeline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# urls = [
#     "https://www.nytimes.com/2020/07/09/health/virus-aerosols-who.html",
#     "https://www.nytimes.com/2020/03/03/travel/how-to-clean-your-airplane-seat-and-space.html",
#     "https://www.nytimes.com/2020/03/11/us/airports-airlines-coronavirus.html",
#     "https://www.nytimes.com/2020/06/27/world/europe/coronavirus-spread-asymptomatic.html",
#     "https://www.nytimes.com/2020/06/01/business/coronavirus-airports-airlines.html",
#     "https://www.nytimes.com/2020/07/21/travel/crowded-flights-coronavirus.html",
#     "https://www.nytimes.com/2020/08/20/travel/airplanes-coronavirus.html",
#     "https://www.nytimes.com/2020/03/19/health/coronavirus-travel-ban.html",
#     "https://www.nytimes.com/2020/01/17/health/china-coronavirus-airport-screening.html",
#     "https://www.nytimes.com/2020/03/11/travel/airports-coronavirus.html",
#     "https://www.nytimes.com/2020/08/05/world/europe/germany-coronavirus-test-travelers.html",
#     "https://www.nytimes.com/2020/05/27/travel/is-flying-safe-coronavirus.html",
#     "https://www.nytimes.com/2020/08/18/business/airport-remodeling-coronavirus-safety.html"
#         ]
"""
query_airport = "aerosol viral transmission on airplane"
query_school = "aerosol viral transmission in school"
query_office = "aerosol viral transmission in office"
corpus = ["The virus spreads very efficiently in closed rooms.",
          "The virus has an efficient aerosol transmission in airplanes.",
          "Crowded places are prone to quick spread of the virus.",
          "Places where individuals cannot keep distance are dangerous.",
          "Kids in school are exposed to viral thread and transmission.",
          "People in office should definitely wear face masks all the time.",
          "In airplanes the air ventilation is optimal for unhindered spread of the virus.",
          "Managers are instructed to have their workforce work from home.",
          "There a lot of cases of infection in air places.",
          "Places without good air circulation susceptible to high risk of infection.",
          "Transmission at home is comparably low.",
          "Large family fests are a risk to the elderly.",
          "Garden parties yield a low risk for spreading the virus."]
emb_corpus = get_embeddings_from_corpus(corpus)
emb_query_airport = get_embeddings_from_corpus(query_airport)
emb_query_school = get_embeddings_from_corpus(query_school)
emb_query_office = get_embeddings_from_corpus(query_office)
results = filter(emb_corpus, emb_query_airport, 8)
print("Top results are:")
for i, index in enumerate(results):
    if i>4: continue
    print(f"Nr. {i+1}: {corpus[index]}")

print("After reranking against concept 'airplane':")
reranked = post_filter(emb_corpus, en("airplane"), results, 5)
for i, index in enumerate(reranked):
    print(f"Nr. {i+1}: {corpus[index]}")


# embed topics and a searcher and see where they sit
context_topics = ["school", "airplane", "airport", "university", "office"]
emb_context_topics = get_embeddings_from_corpus(context_topics)
searcher = torch.mean(emb_context_topics, dim=0, keepdim=True)

# map query against closest_topic and move searcher in dir of closest_topic
ind = filter(emb_context_topics, emb_query_airport, 1)[0]
closest_topic_emb = emb_context_topics[ind].reshape(1, 768)
dir =  closest_topic_emb - searcher
new_searcher = searcher + dir * 0.5


pca = PCA(n_components=2)
emb = torch.cat((emb_context_topics, searcher, emb_query_airport, emb_query_school, emb_query_office, new_searcher), dim=0)
pc = pca.fit_transform(emb.cpu())
x, y = pc[:,0], pc[:,1]
fig, ax = plt.subplots()
for i in range(len(context_topics) + 5):
    if i == len(context_topics) + 4: # this is the new searcher
        ax.scatter(x[i], y[i], c='r', marker='o')
    if i < len(context_topics):
        ax.scatter(x[i], y[i], c='r', marker='o')
    elif i == len(context_topics):  # this is the searcher
        ax.scatter(x[i], y[i], c='g', marker='o')
    elif i >= len(context_topics) + 1: # this is the queries
        ax.scatter(x[i], y[i], c='b', marker='o')


for i, txt in enumerate(context_topics + ["searcher", "airplane query", "school query", "office query", "new searcher"]):
    ax.annotate(txt, (x[i] + 0.25, y[i] + 0.25))

"""

"""
Plot mitigation measures and impacts
"""

measures = ["hand washing", "face masks", "social distance", "air filtration", "surface cleaning"] # health
impacts = ["stock market crash", "recession", "financial market", "aviation industry", "food industry", "meat industry", "restaurant industry", "retail", "tourism", # economic
           "health",
           "culture", # culture
           "society",
           "politics"]
emb_measures = get_embeddings_from_corpus(measures)
emb_impacts = get_embeddings_from_corpus(impacts)

pc = PCA(n_components=2).fit_transform(torch.cat((emb_measures, emb_impacts), dim=0).cpu())
x, y = pc[:,0], pc[:,1]
fig, ax = plt.subplots()
for i, measure in enumerate(measures):
    ax.scatter(x[i], y[i], c='r', marker='o')
    ax.annotate(measure, (x[i] + 0.25, y[i] + 0.25))
for j, impact in enumerate(impacts):
    ax.scatter(x[len(measures) + j], y[len(measures) + j], c='b', marker='o')
    ax.annotate(impact, (x[len(measures) + j] + 0.25, y[len(measures) + j] + 0.25))
plt.show()

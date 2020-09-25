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
"""
measures = ["hand washing", "face masks", "social distance", "air filtration", "surface cleaning"]
impact_economic = ["stock market crash", "recession", "financial market", "aviation industry", "food industry", "meat industry", "restaurant industry", "retail", "tourism"]
impact_culture = ["cinema", "education", "sports", "television", "arts", "music", "fashion", "performing arts", "video games industry"]
impact_society = ["religion", "gender", "human rights", "healthcare workers", "strikes", "social", "mental health", "racism", "public transport"]
# impact_politics = ["european union", "internation relations", "legislation", "national responses", "protests", "United Nations response", "World Healt Organization response"]
impacts = impact_economic + impact_culture + impact_society + impact_society# + impact_politics
emb_measures = get_embeddings_from_corpus(measures)
emb_impacts = get_embeddings_from_corpus(impacts)
searcher = torch.mean(emb_measures, dim=0, keepdim=True)
# pc = PCA(n_components=2).fit_transform(torch.cat((emb_measures, emb_impacts), dim=0).cpu())
pc = PCA(n_components=2).fit_transform(torch.cat((emb_measures, searcher), dim=0).cpu().cpu())
x, y = pc[:,0], pc[:,1]
fig, ax = plt.subplots()
for i, action in enumerate(measures):#+impacts):
    if action in measures:
        color = "r"
    elif action in impact_economic:
        color = "b"
    elif action in impact_culture:
        color = "g"
    elif action in impact_society:
        color = "y"
    # elif action in impact_politics:
    #     color = "black"
    ax.scatter(x[i], y[i], c=color, marker='o')
    ax.annotate(action, (x[i] - 0.5, y[i] + 0.35))
ax.scatter(x[len(measures)], y[len(measures)], c='g', marker='o')
ax.annotate("searcher", (x[len(measures)] - 0.5, y[len(measures)] + 0.35))
plt.show()

"""

def geometric(number: int) -> float:
    return (1/1.3)**number

def step(searcher: Tensor,
         top_indices: List[int],
         emb_measures: Tensor,
         counter: int
         ) -> Tuple[Tensor, Tensor]:
    """

    :param searcher:
    :param top_indices:
    :param emb_measures:
    :param counter:
    :return:
    """
    topics = [emb_measures[filter(emb_measures, emb_corpus[index], 1)[0]].reshape(1,768) for index in top_indices]
    searcher_history = searcher.clone()
    for topic in topics:
        direction = topic - searcher
        distance = geometric(counter) / len(topics)
        searcher = searcher + (direction * distance)
        searcher_history = torch.cat((searcher_history, searcher), dim=0)
    return searcher, searcher_history

urls = ["https://www.nytimes.com/2020/06/02/realestate/virus-social-distancing-etiquette-rules.html",
        "https://www.nytimes.com/2020/05/04/us/social-distancing-rules-coronavirus.html",
        "https://www.nytimes.com/2020/03/16/smarter-living/coronavirus-social-distancing.html",
        "https://www.nytimes.com/2020/06/18/nyregion/coronavirus-ny-social-distancing.html",
        "https://www.nytimes.com/2020/03/18/world/clean-home-coronavirus.html",
        "https://www.nytimes.com/2020/05/06/well/live/coronavirus-cleaning-cleaners-disinfectants-home.html",
        "https://www.nytimes.com/guides/smarterliving/how-to-clean",
        "https://www.nytimes.com/wirecutter/reviews/best-all-purpose-cleaner/",
        "https://www.nytimes.com/2020/03/13/world/how-to-wash-your-hands-coronavirus.html",
        "https://www.nytimes.com/2020/06/11/well/live/the-hand-washing-wars.html",
        "https://www.nytimes.com/interactive/2020/04/15/burst/how-to-wash-your-hands.html",
        "https://www.nytimes.com/2016/04/21/health/washing-hands.html",
        "https://www.nytimes.com/wirecutter/reviews/best-cloth-face-masks/",
        "https://www.nytimes.com/interactive/2020/07/17/upshot/coronavirus-face-mask-map.html",
        "https://www.nytimes.com/article/face-shield-mask-california-coronavirus.html",
        "https://www.nytimes.com/interactive/2020/08/10/nyregion/nyc-subway-coronavirus.html",
        "https://www.nytimes.com/2020/03/04/opinion/coronavirus-buildings.html",
        "https://www.nytimes.com/2020/07/27/health/coronavirus-mask-protection.html",
        "https://www.nytimes.com/2020/07/06/health/coronavirus-airborne-aerosols.html",
        "https://www.nytimes.com/2020/07/30/opinion/coronavirus-aerosols.html"]

colors = ["#00FFCC", "#00FFCC", "#00FFCC", "#00FFCC", # social distancing
          "#F7347A", "#F7347A", "#F7347A", "#F7347A", # surface cleaning
          "#FFFF66", "#FFFF66", "#FFFF66", "#FFFF66", # washing hands
          "#F08080", "#F08080", "#F08080", "#F08080", # face masks
          "#990000", "#990000", "#990000", "#990000"] # air circulation

# Embed context, corpus and place searcher
measures = ["social distance", "surface cleaning", "hand washing", "face masks", "air filtration"]
titles, corpus, emb_corpus = get_corpus_from_links(urls)
emb_measures = get_embeddings_from_corpus(measures)
searcher = torch.mean(emb_measures, dim=0, keepdim=True)

# We assume the user is interested in hand washing
query = "Is hand washing an effective measure against the virus?"
emb_query = get_embeddings_from_corpus(query)
top_indices = filter(emb_corpus, emb_query, 5)
# the top indices are [8, 11, 4, 10, 5]
# we can see that 8, 11, 10 are relevant and 4, 5 are not



searcher, history = step(searcher, top_indices[:2], emb_measures, 1)
searcher, history2 = step(searcher, top_indices[:2], emb_measures, 2)
history2 = history2[1:]
p = 0
pc = PCA(n_components=2).fit_transform(torch.cat((emb_measures, emb_corpus, history, history2), dim=0).cpu())
x, y = pc[:,0], pc[:,1]
fig, ax = plt.subplots()
for i in range(len(corpus) + len(measures) + history.size()[0] + history2.size()[0]):
    if i < len(measures): # measures
        ax.scatter(x[i], y[i], c=colors[i * (len(measures)-1)], marker='o')
        ax.annotate(measures[i], (x[i] - 3.5, y[i] + 0.5))
    elif i < len(measures) + len(corpus): # docs
        ax.scatter(x[i], y[i], c=colors[i - len(measures)], marker='o')
        name = str(i-len(measures)+1) if i-len(measures) >= 10 else " " + str(i-len(measures)+1)
        ax.annotate(name, (x[i]-0.2, y[i]-0.2), fontsize=6)
    else: # searcher
        continue
        ax.scatter(x[i], y[i], c='b', marker='o')
        ax.annotate(str(p), (x[i] + 0.5, y[i] - 0.25 ))
        p += 1
plt.axis('off')
plt.show()
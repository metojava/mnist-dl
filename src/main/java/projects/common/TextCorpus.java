package projects.common;

import java.util.List;
import java.util.Map;

/** Embedded mini-corpora so every NLP project runs offline. */
public final class TextCorpus {
    private TextCorpus() {}

    public static final List<String> POSITIVE = List.of(
        "this movie was absolutely wonderful and moving",
        "a brilliant film with superb acting and a great story",
        "loved every minute of it truly delightful",
        "one of the best films i have seen this year",
        "fantastic direction and a powerful performance",
        "heartwarming funny and beautifully shot",
        "an instant classic i enjoyed it thoroughly",
        "the cast was excellent and the plot kept me hooked",
        "wonderful storytelling with memorable characters",
        "amazing visuals and a perfect soundtrack",
        "i was captivated from start to finish",
        "a joyful experience highly recommend it",
        "great fun for the whole family",
        "the best comedy in years laugh out loud funny",
        "stunning cinematography and brilliant dialogue",
        "an absolute masterpiece of modern cinema",
        "charming witty and full of heart",
        "excellent pacing never a dull moment",
        "superb acting from the entire cast",
        "a feel good movie that delivers pure joy",
        "impressive bold and utterly engaging",
        "the story was touching and the ending perfect",
        "brilliant script executed flawlessly",
        "wonderful performances across the board",
        "i loved it a truly great night at the movies",
        "beautiful moving and unforgettable",
        "a delightful surprise packed with charm",
        "outstanding in every way possible",
        "terrific action scenes and strong characters",
        "genuinely funny warm and clever",
        "the film exceeded all my expectations",
        "magnificent a triumph of filmmaking",
        "enjoyable from the first frame to the last",
        "a wonderful tale told with skill and passion",
        "fresh inventive and thoroughly entertaining",
        "top notch production values and great music",
        "smart sharp and satisfying",
        "an excellent addition to the genre",
        "loved the characters and the heartfelt story",
        "absolutely fantastic would watch again");

    public static final List<String> NEGATIVE = List.of(
        "this movie was terrible and painfully boring",
        "a complete waste of time and money",
        "awful acting and a boring predictable plot",
        "one of the worst films i have ever suffered through",
        "poorly directed and utterly forgettable",
        "dull slow and completely joyless",
        "a disaster from start to finish",
        "the cast seemed lost and the plot made no sense",
        "weak writing ruined an interesting premise",
        "ugly visuals and an annoying soundtrack",
        "i could not wait for it to end",
        "a tedious mess avoid at all costs",
        "not fun at all painful to sit through",
        "the worst comedy in years not even slightly funny",
        "murky cinematography and clunky dialogue",
        "an absolute failure of storytelling",
        "lifeless cold and forgettable",
        "terrible pacing every scene dragged",
        "wooden acting from everyone on screen",
        "a depressing chore of a movie",
        "uninspired lazy and boring",
        "the story was incoherent and the ending awful",
        "a bad script badly executed",
        "poor performances all around",
        "i hated it a complete waste of an evening",
        "ugly mean spirited and instantly forgettable",
        "a disappointment from beginning to end",
        "awful in every conceivable way",
        "laughable action scenes and hollow characters",
        "neither funny nor clever just dull",
        "the film fell far below my expectations",
        "a miserable failure of a film",
        "unwatchable from the first frame",
        "a tired retelling with no passion",
        "stale derivative and thoroughly boring",
        "cheap looking with terrible music",
        "shallow pointless and unsatisfying",
        "a poor excuse for a movie",
        "hated the characters and the contrived story",
        "absolutely terrible would never watch again");

    public static final Map<String, List<String>> HEADLINES = Map.of(
        "sports", List.of(
            "lakers win championship in overtime thriller",
            "striker scores hat trick in world cup qualifier",
            "olympic swimmer breaks world record final",
            "tennis star returns to grand slam victory",
            "quarterback signs record contract extension",
            "united defeats rivals in derby classic",
            "marathon champion targets new personal best",
            "coach announces retirement after title run",
            "rookie pitcher throws no hitter debut",
            "boxing title fight ends dramatic knockout",
            "cyclist wins tour stage mountain finish",
            "goalkeeper penalty saves clinch cup"),
        "politics", List.of(
            "senate passes sweeping infrastructure bill",
            "president signs executive order climate policy",
            "election results spark coalition talks",
            "lawmakers debate tax reform package",
            "diplomats meet summit trade agreement",
            "parliament votes budget austerity plan",
            "governor declares emergency relief funding",
            "minister resigns amid corruption probe",
            "voters approve ballot minimum wage",
            "cabinet reshuffle follows polling slump",
            "congress hearings examine campaign finance",
            "mayor unveils housing affordability plan"),
        "technology", List.of(
            "new ai model beats benchmark language tasks",
            "startup unveils foldable smartphone display",
            "quantum computer achieves error correction milestone",
            "chipmaker launches faster gpu datacenter",
            "open source database release boosts performance",
            "autonomous vehicle completes cross country trial",
            "battery breakthrough promises longer range evs",
            "cybersecurity firm patches critical zero day",
            "satellite internet service expands coverage",
            "robot learns dexterous manipulation tasks",
            "cloud provider cuts pricing enterprise plans",
            "researchers demo brain computer interface typing"),
        "business", List.of(
            "stocks rally as fed signals rate pause",
            "retail giant reports record quarterly earnings",
            "merger creates largest semiconductor company",
            "oil prices climb supply concerns persist",
            "bank raises growth forecast global economy",
            "airline orders hundreds fuel efficient jets",
            "housing market cools mortgage rates rise",
            "electric car maker expands factories europe",
            "inflation eases consumer prices steady",
            "venture funding rebounds startup ecosystem",
            "currency weakens export outlook improves",
            "supermarket chain launches delivery partnership"));

    public static final Map<String, List<String>> LANG_SENTENCES = Map.of(
        "english", List.of(
            "the quick brown fox jumps over the lazy dog",
            "where is the nearest train station please",
            "i would like a cup of coffee and some bread",
            "she enjoys reading books on rainy afternoons",
            "our meeting starts at nine in the morning",
            "the weather is beautiful today is it not",
            "he plays football with his friends weekly",
            "this city has wonderful museums and parks",
            "can you help me find my luggage",
            "breakfast is served between seven and ten",
            "learning new languages opens many doors",
            "the children are playing in the garden",
            "we traveled north to see the mountains",
            "thank you very much for your kindness"),
        "french", List.of(
            "le renard brun rapide saute par dessus le chien paresseux",
            "ou est la gare la plus proche s il vous plait",
            "je voudrais une tasse de cafe et du pain",
            "elle aime lire des livres les apres midi de pluie",
            "notre reunion commence a neuf heures du matin",
            "le temps est magnifique aujourd hui n est ce pas",
            "il joue au football avec ses amis chaque semaine",
            "cette ville possede des musees merveilleux",
            "pouvez vous m aider a trouver mes bagages",
            "le petit dejeuner est servi entre sept et dix heures",
            "apprendre des langues ouvre beaucoup de portes",
            "les enfants jouent dans le jardin",
            "nous avons voyage vers le nord pour voir les montagnes",
            "merci beaucoup pour votre gentillesse"),
        "german", List.of(
            "der schnelle braune fuchs springt ueber den faulen hund",
            "wo ist bitte der naechste bahnhof",
            "ich moechte eine tasse kaffee und etwas brot",
            "sie liest gern buecher an regnerischen nachmittagen",
            "unsere besprechung beginnt um neun uhr morgens",
            "das wetter ist heute schoen nicht wahr",
            "er spielt jede woche fussball mit seinen freunden",
            "diese stadt hat wunderbare museen und parkanlagen",
            "koennen sie mir helfen mein gepaeck zu finden",
            "das fruehstueck wird zwischen sieben und zehn serviert",
            "sprachen lernen oeffnet viele tueren",
            "die kinder spielen im garten",
            "wir sind nach norden gereist um die berge zu sehen",
            "vielen dank fuer ihre freundlichkeit"),
        "spanish", List.of(
            "el rapido zorro marron salta sobre el perro perezoso",
            "donde esta la estacion de tren mas cercana por favor",
            "quisiera una taza de cafe y un poco de pan",
            "ella disfruta leyendo libros las tardes lluviosas",
            "nuestra reunion comienza a las nueve de la manana",
            "el clima esta hermoso hoy no es cierto",
            "el juega futbol con sus amigos semanalmente",
            "esta ciudad tiene museos maravillosos y parques",
            "puede ayudarme a encontrar mi equipaje",
            "el desayuno se sirve entre las siete y las diez",
            "aprender idiomas abre muchas puertas",
            "los ninos estan jugando en el jardin",
            "viajamos al norte para ver las montanas",
            "muchas gracias por su amabilidad"),
        "italian", List.of(
            "la volpe marrone veloce salta sopra il cane pigro",
            "dov e la stazione ferroviaria piu vicina per favore",
            "vorrei una tazza di caffe e un po di pane",
            "le piace leggere libri nei pomeriggi piovosi",
            "la nostra riunione inizia alle nove del mattino",
            "il tempo e bellissimo oggi non e vero",
            "gioca a calcio con gli amici ogni settimana",
            "questa citta ha musei meravigliosi e parchi",
            "puo aiutarmi a trovare i miei bagagli",
            "la colazione e servita tra le sette e le dieci",
            "imparare le lingue apre molte porte",
            "i bambini stanno giocando nel giardino",
            "abbiamo viaggiato a nord per vedere le montagne",
            "grazie mille per la sua gentilezza"));

    /** Public-domain-style excerpt used by char-LSTM projects. */
    public static final String SHAKESPEARE_EXCERPT = """
        To be, or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them. To die, to sleep-
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to, 'tis a consummation
        Devoutly to be wish'd. To die, to sleep;
        To sleep: perchance to dream: ay, there's the rub;
        For in that sleep of death what dreams may come
        When we have shuffled off this mortal coil,
        Must give us pause: there's the respect
        That makes calamity of so long life;
        For who would bear the whips and scorns of time,
        The oppressor's wrong, the proud man's contumely,
        The pangs of despised love, the law's delay,
        The insolence of office and the spurns
        That patient merit of the unworthy takes,
        When he himself might his quietus make
        With a bare bodkin? who would fardels bear,
        To grunt and sweat under a weary life,
        But that the dread of something after death,
        The undiscover'd country from whose bourn
        No traveller returns, puzzles the will
        And makes us rather bear those ills we have
        Than fly to others that we know not of?
        Thus conscience does make cowards of us all;
        And thus the native hue of resolution
        Is sicklied o'er with the pale cast of thought,
        And enterprises of great pitch and moment
        With this regard their currents turn awry,
        And lose the name of action. Soft you now!
        The fair Ophelia! Nymph, in thy orisons
        Be all my sins remember'd.
        """;

    public static final Map<String, List<String>> INTENTS = Map.of(
        "greeting", List.of(
            "hello there", "hi how are you", "good morning", "hey what's up", "hiya",
            "good evening", "hello bot", "greetings", "nice to meet you", "howdy"),
        "goodbye", List.of(
            "bye bye", "see you later", "goodbye for now", "talk to you tomorrow", "catch you later",
            "good night", "farewell", "i have to go", "later friend", "bye now"),
        "weather", List.of(
            "what is the weather today", "will it rain tomorrow", "is it sunny outside",
            "temperature right now", "forecast for the weekend", "chance of rain tonight",
            "how windy is it", "is snow expected", "humidity level outside", "weather this week"),
        "ordering_food", List.of(
            "i want to order pizza", "can i get a burger delivered", "menu for tonight please",
            "book a table for two", "add fries to my order", "what restaurants deliver sushi",
            "order takeout thai food", "reserve dinner reservation seven pm", "price of the lunch special",
            "track my food delivery"),
        "help", List.of(
            "i need assistance", "can you help me", "support please", "something is broken",
            "how do i use this feature", "give me instructions", "explain how it works",
            "technical support needed", "guide me through setup", "answer my question please"),
        "thanks", List.of(
            "thank you so much", "thanks a lot", "much appreciated", "that was helpful thanks",
            "great job thank you", "thanks for the info", "you saved me time thanks",
            "appreciate your help", "many thanks friend", "ty very much"));
}

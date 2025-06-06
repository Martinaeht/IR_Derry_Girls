# Indices from BM25/aligned with FAISS indices

queries = {
    "What's that friendship programm called they take part in?": [11, 14],
    # A bloody 'Friends Across the Barricades' thing.
    # But I didn't meet him at 'Friends Across the Barricades'.
    
    "How often do they talk about bombs?": [17, 35, 323, 1691, 4265, 313],
    # They're bombing the bridge, apparently.
    # Well, I don't know about the rest of you, but I'm not enjoying this bomb.
    # How long does it take to defuse a fecking bomb?
    # Bomb! He has a bomb!
    # The statement has raised hopes for peace and an end to twenty-five years of bombing and shooting that has led to the deaths of more than three-thousand people.
    # This bloody bomb.

    "detonate": [2208],
    # Newscaster: Emergency services are urging anyone with medical training to come to the scene immediately. The device was detonated at 3pm this afternoon.

    "Is the political realm of the Troubles talked about at all?": [39, 1143, 2676],
    # I'm pretty sure interfering with your sunbed sessions isn't very high up on anyone's political agenda, Aunt Sarah.
    # Well, there's actually a political element to it, Katya, and there's a religious element.
    # It's about The Troubles, in a political sense, but also about my own troubles, in a personal sense.

    "What is the name of the school they attend?": [191, 2361, 2627],
    # Sister Michael: James will be the first ever boy to study here at Our Lady Immaculate College.
    # Sister Michael: So, our Lady Immaculate girls have been split into groups, A through to F, as have the Londonderry Boys Academy.
    # Sister Michael: As you all know, at the beginning of term, the Bishop graciously bestowed a beautiful piece of religious art onto Our Lady Immaculate College, the enchanting Child of Prague.

    "What show does Erin never miss?": [91],
    # Orla: But Erin, 'Murder She Wrote' is on tonight. You never miss 'Murder She Wrote'.

    "Where's the class trip going to?": [353, 368, 380, 482, 606],
    # (Sister Michael: Notice from Mr. Macaulay. This year's destination for the Euro Trotters trip will be, dramatic pause… (Everyone waits expectantly.)) Sister Michael: Did you actually want me to do the dramatic pause? Interesting. Ah, Paris. It's going to be Paris.
    # Orla: If we go to Paris, I'd like to meet Nicole. (stares at dummy sweet and licks it)
    # Charlene: Right. Are you signing up for Paris? I can't convince that lot to come. Looks like I'm going to need someone to hang out with.
    # Michelle: Where the fuck are we going to get the money for Paris now?
    # Michelle: So, not only are we not going to Paris, we're spending our Sunday scrubbing Fionnula's fucking fish hole for free.

    "How many bags of chips do they end up ordering?": [411, 419, 432, 470, 537, 598],
    # Gerry: Okay. That's one portion of red fish, one portion of white fish, two bags of chips.
    # Gerry: Okay. That's one portion of red fish, one portion of white fish, two bags of chips. (Granda Joe: No, no, no, no. Two bags won't be enough.)
    # Gerry: Okay. That's one portion of red fish, one portion of white fish, two bags of chips. (Granda Joe: No, no, no, no. Two bags won't be enough. (Joe stands and walks towards table.) Gerry: Two's plenty, Joe. Granda Joe: (carrying baby) Four. Four should cover it.)
    # Gerry: Okay. Four bags of chips then.
    # (Sarah: Will four bags be enough? Gerry: More than enough! Sarah: I say we need five, to be safe. Do you not think, Da? (Granda Joe: Stick down five. Gerry: (sighs in frustration)) Five bags of chips, then!
    # Erin: (reading from paper) Seven bags of chips. Twelve chicken nuggets. One small battered hotdog. One plain chicken burger.
    # Sarah: We should ring UTV, get them to do an interview. When Shauna Sharkey was interviewed, do you mind the time where her and her brother got hijacked, well, Fionnula gave her free chips for a month.
    # Mary: Save your breath, Sarah. There'll be no free chips. (angrily) There'll be no chips, full stop.

    "Is James gay?": [664, 711, 1500, 2124, 2186],
    # Sarah: What, not even the wee gay fella? James: I'm not gay. 
    # (Granda Joe: Look, I know the fella's gay…) James: I'm not gay. (Granda Joe: But gay or not… James: Who said I was gay?)
    # (Deidre: Hope to God it's not the gay thing you're offended by. James: (still looking upset)) There is no gay thing. (Deidre: Because I'd be disappointed in you, Mary. I'll not lie. Mary: Of course not. I mean, if anything the gay thing sort of cancels out the English thing. James: Again, no gay thing.) 
    # James: (holding up newspapers) I support gays, even though I, myself, am not actually gay!
    # (Michelle: Ignore him, Erin. These gays, they all stick together. James: (annoyed)) I'm not gay.

    "What's Erin's dog called?": [688, 717, 718, 729, 741, 890, 894, 948, 1015],
    # Erin: (looking upset) Toto was much more than a dog. Toto was my best friend. (holds photograph up for all to see)
    # Sarah: Ach don't talk to me. I was in bits last night. Didn't even manage my Chinese. Poor… Tonto. 
    # Erin: (offended) Toto. His name was Toto, Aunt Sarah.
    # Erin: Mummy, what happened to Toto, it's just hit me so hard and I'm worried, it might affect my performance.
    # Erin: Doesn't that dog look like Toto?
    # Erin: (scoffs) Right. Maybe. But, as I think we mentioned, Toto's dead. My ma saw him get hit by an army Land Rover and then buried him in our back garden, so probably not.
    # Father Peter: don't you see? Toto was sent back to lead you to that chapel, to that statue. Because you're special Erin. You have been chosen.
    # Erin: Yes, and now he's talking about digging him up, and when he does, he'll realise that Toto has not, in fact, been resurrected. He'll realise that Toto is just dead. Very, very dead. And we've all been talking shite.
    # Erin: The chapel. She was there. That dog was Toto. Oh it's all adding up now, isn't it Mammy? 

    "What's the school's paper called? ": [2070],
    # Erin: (shouting) Fine! Well this issue of The Habit will go down in history! In history, I say! ((Orla looks excited.) Michelle: You seriously need to chill the fuck out. [In Sister Michael's office. A copy of The Habit is placed on her desk, with the headline 'THE SECRET LIFE OF A LESBIAN!' written on the front.] Sister Michael: Not happening. Erin: But Sister Michael- Sister Michael: Don't even think about arguing. I can't be doing with the board of governors getting wind of something like this at the moment. They're a thorn in my arse at the best of times. I mean, for God's sake, Erin, what happened to 'Shoes of the World'? Erin: This is more important. This is about, you know, gay rights. Sister Michael: (rolling her eyes) I see. And I take it that's what the camp shirts are in aid of. A very serious uniform violation, by the way. But let's take one problem at a time, shall we? Erin: (looking offended) This isn't a problem, Sister. (pointing at the newspaper) This is groundbreaking journalism. Sister Michael: (sternly) They are not to be distributed tomorrow. Do I make myself clear? James: But that's censorship. Sister Michael: Well done. You are correct. You're being censored. Now go.)

    "Who does the priest hook up with?": [2413, 1056, 1038],
    # (Father Peter: Okay, so I see a few familiar faces out there. As some of you may know, I took a bit of a sabbatical last year.) Michelle: Do you mean when you shacked up with a slutty hairdresser, but then she dumped you?
    # Father Peter: Okay, I think we should just move on. Sister Michael: The hairdresser certainly did.
    # (Michelle: So he's just going to pack in the priesthood, like completely? Erin: Well you can't exactly go part-time. (Clare looks furious.) Orla: (sucking a lollipop happily) All because of us.) Erin: Not all because of us, Orla. I mean, a bit because of us. But mostly because it turns out he had a connection with one of the colourists in Hair and Flair, who does our Sarah's forwards, by the way. And apparently she's a dirty tramp. 
    # Father Peter: And you know, it became more challenging, when I met this amazing girl.

    "Who spray-painted the sign on the house wall? ": [1838], 
    #The mural on our house, the spray paint. It was Emmett, and I can prove it.
}


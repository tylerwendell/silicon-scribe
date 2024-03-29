from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from language_enum import LanguageEnum

tokenizer =AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang="ell_Grek")
model =  AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

article = """ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος.
οὖτος ἦν ἐν ἀρχῇ πρὸς τὸν θεόν.
πάντα δι᾽ αὐτοῦ ἐγένετο, καὶ χωρὶς αὐτοῦ ἐγένετο οὐδὲ ἕν. ὃ γέγονεν
ἐν αὐτῶ ζωὴ ἦν, καὶ ἡ ζωὴ ἦν τὸ φῶς τῶν ἀνθρώπων·
καὶ τὸ φῶς ἐν τῇ σκοτίᾳ φαίνει, καὶ ἡ σκοτία αὐτὸ οὐ κατέλαβεν.
ἐγένετο ἄνθρωπος ἀπεσταλμένος παρὰ θεοῦ, ὄνομα αὐτῶ ἰωάννης·
οὖτος ἦλθεν εἰς μαρτυρίαν, ἵνα μαρτυρήσῃ περὶ τοῦ φωτός, ἵνα πάντες πιστεύσωσιν δι᾽ αὐτοῦ.
οὐκ ἦν ἐκεῖνος τὸ φῶς, ἀλλ᾽ ἵνα μαρτυρήσῃ περὶ τοῦ φωτός.
ἦν τὸ φῶς τὸ ἀληθινόν, ὃ φωτίζει πάντα ἄνθρωπον, ἐρχόμενον εἰς τὸν κόσμον.
ἐν τῶ κόσμῳ ἦν, καὶ ὁ κόσμος δι᾽ αὐτοῦ ἐγένετο, καὶ ὁ κόσμος αὐτὸν οὐκ ἔγνω.
εἰς τὰ ἴδια ἦλθεν, καὶ οἱ ἴδιοι αὐτὸν οὐ παρέλαβον.
ὅσοι δὲ ἔλαβον αὐτόν, ἔδωκεν αὐτοῖς ἐξουσίαν τέκνα θεοῦ γενέσθαι, τοῖς πιστεύουσιν εἰς τὸ ὄνομα αὐτοῦ,
οἳ οὐκ ἐξ αἱμάτων οὐδὲ ἐκ θελήματος σαρκὸς οὐδὲ ἐκ θελήματος ἀνδρὸς ἀλλ᾽ ἐκ θεοῦ ἐγεννήθησαν.
καὶ ὁ λόγος σὰρξ ἐγένετο καὶ ἐσκήνωσεν ἐν ἡμῖν, καὶ ἐθεασάμεθα τὴν δόξαν αὐτοῦ, δόξαν ὡς μονογενοῦς παρὰ πατρός, πλήρης χάριτος καὶ ἀληθείας.
ἰωάννης μαρτυρεῖ περὶ αὐτοῦ καὶ κέκραγεν λέγων, οὖτος ἦν ὃν εἶπον, ὁ ὀπίσω μου ἐρχόμενος ἔμπροσθέν μου γέγονεν, ὅτι πρῶτός μου ἦν.
ὅτι ἐκ τοῦ πληρώματος αὐτοῦ ἡμεῖς πάντες ἐλάβομεν, καὶ χάριν ἀντὶ χάριτος·
ὅτι ὁ νόμος διὰ μωϊσέως ἐδόθη, ἡ χάρις καὶ ἡ ἀλήθεια διὰ ἰησοῦ χριστοῦ ἐγένετο.
θεὸν οὐδεὶς ἑώρακεν πώποτε· μονογενὴς θεὸς ὁ ὢν εἰς τὸν κόλπον τοῦ πατρὸς ἐκεῖνος ἐξηγήσατο.
καὶ αὕτη ἐστὶν ἡ μαρτυρία τοῦ ἰωάννου, ὅτε ἀπέστειλαν [πρὸς αὐτὸν] οἱ ἰουδαῖοι ἐξ ἱεροσολύμων ἱερεῖς καὶ λευίτας ἵνα ἐρωτήσωσιν αὐτόν, σὺ τίς εἶ;
καὶ ὡμολόγησεν καὶ οὐκ ἠρνήσατο, καὶ ὡμολόγησεν ὅτι ἐγὼ οὐκ εἰμὶ ὁ χριστός.
καὶ ἠρώτησαν αὐτόν, τί οὗν; σύ ἠλίας εἶ; καὶ λέγει, οὐκ εἰμί. ὁ προφήτης εἶ σύ; καὶ ἀπεκρίθη, οὔ.
εἶπαν οὗν αὐτῶ, τίς εἶ; ἵνα ἀπόκρισιν δῶμεν τοῖς πέμψασιν ἡμᾶς· τί λέγεις περὶ σεαυτοῦ;
ἔφη, ἐγὼ φωνὴ βοῶντος ἐν τῇ ἐρήμῳ, εὐθύνατε τὴν ὁδὸν κυρίου, καθὼς εἶπεν ἠσαΐας ὁ προφήτης.
καὶ ἀπεσταλμένοι ἦσαν ἐκ τῶν φαρισαίων.
καὶ ἠρώτησαν αὐτὸν καὶ εἶπαν αὐτῶ, τί οὗν βαπτίζεις εἰ σὺ οὐκ εἶ ὁ χριστὸς οὐδὲ ἠλίας οὐδὲ ὁ προφήτης;
ἀπεκρίθη αὐτοῖς ὁ ἰωάννης λέγων, ἐγὼ βαπτίζω ἐν ὕδατι· μέσος ὑμῶν ἕστηκεν ὃν ὑμεῖς οὐκ οἴδατε,
ὁ ὀπίσω μου ἐρχόμενος, οὖ οὐκ εἰμὶ [ἐγὼ] ἄξιος ἵνα λύσω αὐτοῦ τὸν ἱμάντα τοῦ ὑποδήματος.
ταῦτα ἐν βηθανίᾳ ἐγένετο πέραν τοῦ ἰορδάνου, ὅπου ἦν ὁ ἰωάννης βαπτίζων.
τῇ ἐπαύριον βλέπει τὸν ἰησοῦν ἐρχόμενον πρὸς αὐτόν, καὶ λέγει, ἴδε ὁ ἀμνὸς τοῦ θεοῦ ὁ αἴρων τὴν ἁμαρτίαν τοῦ κόσμου.
οὖτός ἐστιν ὑπὲρ οὖ ἐγὼ εἶπον, ὀπίσω μου ἔρχεται ἀνὴρ ὃς ἔμπροσθέν μου γέγονεν, ὅτι πρῶτός μου ἦν.
κἀγὼ οὐκ ᾔδειν αὐτόν, ἀλλ᾽ ἵνα φανερωθῇ τῶ ἰσραὴλ διὰ τοῦτο ἦλθον ἐγὼ ἐν ὕδατι βαπτίζων.
καὶ ἐμαρτύρησεν ἰωάννης λέγων ὅτι τεθέαμαι τὸ πνεῦμα καταβαῖνον ὡς περιστερὰν ἐξ οὐρανοῦ, καὶ ἔμεινεν ἐπ᾽ αὐτόν·
κἀγὼ οὐκ ᾔδειν αὐτόν, ἀλλ᾽ ὁ πέμψας με βαπτίζειν ἐν ὕδατι ἐκεῖνός μοι εἶπεν, ἐφ᾽ ὃν ἂν ἴδῃς τὸ πνεῦμα καταβαῖνον καὶ μένον ἐπ᾽ αὐτόν, οὖτός ἐστιν ὁ βαπτίζων ἐν πνεύματι ἁγίῳ.
κἀγὼ ἑώρακα, καὶ μεμαρτύρηκα ὅτι οὖτός ἐστιν ὁ υἱὸς τοῦ θεοῦ.
τῇ ἐπαύριον πάλιν εἱστήκει ὁ ἰωάννης καὶ ἐκ τῶν μαθητῶν αὐτοῦ δύο,
καὶ ἐμβλέψας τῶ ἰησοῦ περιπατοῦντι λέγει, ἴδε ὁ ἀμνὸς τοῦ θεοῦ.
καὶ ἤκουσαν οἱ δύο μαθηταὶ αὐτοῦ λαλοῦντος καὶ ἠκολούθησαν τῶ ἰησοῦ.
στραφεὶς δὲ ὁ ἰησοῦς καὶ θεασάμενος αὐτοὺς ἀκολουθοῦντας λέγει αὐτοῖς, τί ζητεῖτε; οἱ δὲ εἶπαν αὐτῶ, ῥαββί ὃ λέγεται μεθερμηνευόμενον διδάσκαλε, ποῦ μένεις;
λέγει αὐτοῖς, ἔρχεσθε καὶ ὄψεσθε. ἦλθαν οὗν καὶ εἶδαν ποῦ μένει, καὶ παρ᾽ αὐτῶ ἔμειναν τὴν ἡμέραν ἐκείνην· ὥρα ἦν ὡς δεκάτη.
ἦν ἀνδρέας ὁ ἀδελφὸς σίμωνος πέτρου εἷς ἐκ τῶν δύο τῶν ἀκουσάντων παρὰ ἰωάννου καὶ ἀκολουθησάντων αὐτῶ·
εὑρίσκει οὖτος πρῶτον τὸν ἀδελφὸν τὸν ἴδιον σίμωνα καὶ λέγει αὐτῶ, εὑρήκαμεν τὸν μεσσίαν ὅ ἐστιν μεθερμηνευόμενον χριστός·
ἤγαγεν αὐτὸν πρὸς τὸν ἰησοῦν. ἐμβλέψας αὐτῶ ὁ ἰησοῦς εἶπεν, σὺ εἶ σίμων ὁ υἱὸς ἰωάννου· σὺ κληθήσῃ κηφᾶς ὃ ἑρμηνεύεται πέτρος.
τῇ ἐπαύριον ἠθέλησεν ἐξελθεῖν εἰς τὴν γαλιλαίαν, καὶ εὑρίσκει φίλιππον. καὶ λέγει αὐτῶ ὁ ἰησοῦς, ἀκολούθει μοι.
ἦν δὲ ὁ φίλιππος ἀπὸ βηθσαϊδά, ἐκ τῆς πόλεως ἀνδρέου καὶ πέτρου.
εὑρίσκει φίλιππος τὸν ναθαναὴλ καὶ λέγει αὐτῶ, ὃν ἔγραψεν μωϊσῆς ἐν τῶ νόμῳ καὶ οἱ προφῆται εὑρήκαμεν, ἰησοῦν υἱὸν τοῦ ἰωσὴφ τὸν ἀπὸ ναζαρέτ.
καὶ εἶπεν αὐτῶ ναθαναήλ, ἐκ ναζαρὲτ δύναταί τι ἀγαθὸν εἶναι; λέγει αὐτῶ [ὁ] φίλιππος, ἔρχου καὶ ἴδε.
εἶδεν ὁ ἰησοῦς τὸν ναθαναὴλ ἐρχόμενον πρὸς αὐτὸν καὶ λέγει περὶ αὐτοῦ, ἴδε ἀληθῶς ἰσραηλίτης ἐν ᾧ δόλος οὐκ ἔστιν.
λέγει αὐτῶ ναθαναήλ, πόθεν με γινώσκεις; ἀπεκρίθη ἰησοῦς καὶ εἶπεν αὐτῶ, πρὸ τοῦ σε φίλιππον φωνῆσαι ὄντα ὑπὸ τὴν συκῆν εἶδόν σε.
ἀπεκρίθη αὐτῶ ναθαναήλ, ῥαββί, σὺ εἶ ὁ υἱὸς τοῦ θεοῦ, σὺ βασιλεὺς εἶ τοῦ ἰσραήλ.
ἀπεκρίθη ἰησοῦς καὶ εἶπεν αὐτῶ, ὅτι εἶπόν σοι ὅτι εἶδόν σε ὑποκάτω τῆς συκῆς πιστεύεις; μείζω τούτων ὄψῃ.
καὶ λέγει αὐτῶ, ἀμὴν ἀμὴν λέγω ὑμῖν, ὄψεσθε τὸν οὐρανὸν ἀνεῳγότα καὶ τοὺς ἀγγέλους τοῦ θεοῦ ἀναβαίνοντας καὶ καταβαίνοντας ἐπὶ τὸν υἱὸν τοῦ ἀνθρώπου."""

inputs = tokenizer(article, return_tensors="pt")
print("Input Length {}\n".format(len(inputs)))
translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_new_tokens=500)
print("Token Length {}\n".format(len(translated_tokens)))
answer = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

print(answer)

for item in answer:
    print("One of the items: {} \n".format(item))
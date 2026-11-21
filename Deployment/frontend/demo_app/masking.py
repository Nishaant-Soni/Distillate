"""Masking for displayed text: offensive words keep only their first and last letters.

The Try-it page shows text masked by default, so a shared screen (Expo, a recording) never shows
an offensive word in full. Only listed words are masked; the rest stays readable.

The list is kept as SHA-256 hashes, as in ``Model-Pipeline/src/sensitivity/attributions.py``
(whose ``MASKED_TERMS`` are included here), so the repository holds no offensive words. Each
line's comment is the word as displayed. Matching undoes the obfuscation tricks the data
pipeline tests against (``Data-Pipeline/scripts/obfuscate.py``): digit substitution
(``1d10t``) and Cyrillic look-alikes. To add a word:
``hashlib.sha256(word.lower().encode()).hexdigest()``.
"""

from __future__ import annotations

import hashlib
import re
import string

MASKED_TERMS = frozenset({
    # From Model-Pipeline/src/sensitivity/attributions.py (slurs).
    "120f6e5b4ea32f65bda68452fcfaaef06b0136e1d0e4a6f60bc3771fa0936dd6",   # n****r
    "5b3ae48be122f7ed19b4cc587649f41f9d2565df51cfa332f8e7806f4ebb9032",   # n*****s
    "08a841e996781e9e77d30a4e4420a8f501a280b00624e6d1224bf54aaff73eba",   # n***a
    "341d56384afc0f47b34ca18273e793be555507a49444c30d3d0588688de46cb3",   # n****s
    "8f5083e3e5c7dc8932f2bf58212f963f3a44752618c96297f82623f736c52738",   # f****t
    "1e02eec4f1095143be282056557034d4e8ab915342c1af508223141d472d2347",   # f*****s
    "9915ba2d822280f22c283df4e76584a40e0119fc58f73c5f84d4fdb04d04fa6f",   # f**
    "17bde8b4064612e266eaa1b5b1dfa054d1e6dd263f66b392159006d2c9a8ae8a",   # f**s
    "158869a97379229b7681efae9d7f9c9214134e836d649ba53477c0c111414d59",   # r****d
    "c1cabb6f6e431f9c4dea4c3b3264d6fe3829241d674f3496a2dfff6658f0363e",   # r*****s
    "c3de533e9b7fe63b79f648687a30d2861edd92fe7c3cd1f2c485e0a605367624",   # k**e
    "268651b3ece980102f18871fde07189372961e056f858ca147a28d004f876b03",   # k***s
    "98b52c4b6b7d1f48e7477a5ccc10955dd195d0ac5a38c8281bfeb08762634909",   # s**c
    "044eb98b18769b887d5a0258f675e413058b8e9fc9b9786b418bfc6f03f26c98",   # s***s
    "f9d0d9b18ae9033a5ea36df19bf279b059e887a9ae785db81117bceaecc95933",   # c***k
    "45cd3e1bb472d8e285a160a95f6409ec5ce86102e9caaaf0b63fc11db04ab559",   # c****s
    "16ea09fc78ca83ca502cbcf2377acdf280bf18f61e259153f0868405eedab5ef",   # t****y
    "402f7ebd98864afed4817ff4718d357dbfa95458aa4236d6c5959536956fa4e5",   # t******s
    "886d51e97ad7931d0d2af8439ca6d9e4887e3c2b469ed247cbd68ceb3649ccde",   # d**e
    "92a9bf818e7e00021b6d6959f877f5df11b0f989e486db38f35853ebbd552086",   # d***s
    "eef3bd091670c3447022d619c06ad15de96da72b5a66f28bb8b75d1b1c12a05f",   # w*****k
    "0ce875d620076b533c6e681abaa0dc5d9a941366bb000a02ec5351f341b96c61",   # w******s
    "e7b98c6aa5b944e0b315d350d423f895ac9e44fb84f1534b18c2572370a67b9e",   # c**n
    "daec0c235d3135fc1c60aee787f784c59c4d53b2041ff2bacd0d3bebda7246a8",   # c***s
    "cc02032349c833ac5e97bac094560ed40e09acf34cb1978ab7a9840b9bf15b4d",   # g**k
    "82159dd02870c13ec30e4f146db7c317aca568469d0e7e22b8610a2eaa3d3924",   # g***s
    "333f7618092958c75b8c5af6f1ec77b42803922a0fc6ff1570a8af3a3aab3b4a",   # t*******d
    "da0a6b1b9213d48ff10cc45733cc8a925dbd2afb26320ed488ac2efa9a8ae33a",   # t********s
    "22fc75e65a0e9d34324092a7c6a8dba961853294abca4e5914e60c550f48e0c2",   # r*****d
    "17a110e5332848721f27510e1e2bca710a321663bb8a5632e0e46a820fc9e28a",   # r******s
    # Common insults and profanity, for the presets and typical demo input.
    "48fab7a56479b7ec15bd571024dee1cdfe85678289c5cca3ed95769793797c6e",   # i***t
    "50afe4884548838b9ec6ba555fb6e0be1b01f4df387c59b26643c54c5177386c",   # i****s
    "c2d214a484e9d44d34c0c26cca27b34d0eca755d7d7341549885efc7ef592425",   # s****d
    "110c6c5a86fa0de59f9170494cb9841e767243e1f611bb6c513b86e4641284c8",   # m***n
    "a360455901adad1219c78e08d815199a70b234c55e00171f73a378035ee75411",   # m****s
    "ac996ae492f9a987c84f109f2118d3f6632e2259455d30c455a9ecd66c3c4959",   # d**b
    "e9143a29033d9145afc1854ad835b976814bf560b245685a801450f52c288aec",   # d**n
    "0ebdc3317b75839f643387d783535adc360ca01f33c75f7c1e7373adcd675c0b",   # h**l
    "2189c0ed714f0c54ea91fc1d8355e3b1d68723a4fb580e99b088651c80a50f26",   # c**p
    "85fc17f7069acd39a5c636cd0a6530651096128da447959f5e250824857dc559",   # s**t
    "986c99004163a6c5155bbd2bfadc37bee9e279b73c669211f432910425b8e6dd",   # s****y
    "6ac3c336e4094835293a3fed8a4b5fedde1b5e2626d9838fed50693bba00af0e",   # f**k
    "31506a8448a761a448a08aa69d9116ea8a6cb1c6b3f4244b3043051f69c9cc3c",   # f*****g
    "b690cdbe59f14b3be083db03237e04051abd85c4ffe35180723e9cb45cd3a519",   # f****d
    "2062f80093066633876b542212c496501a5e79523cc4ea9b28667dff065afd8f",   # a**
    "f50c51ed2315dcf3fa88181cf033f8029cac64f7dea4048327ca032ec102ea74",   # a*****e
    "d75a838dc758ba17f28bd8dbac605cb70c35465263d5733164521de2f7ef7926",   # b***h
    "e512a05583448f44790783f986b1f36925c8cfc42338ca0e1caa637755bd15ae",   # b*****d
    "ad505b0be8a49b89273e307106fa42133cbd804456724c5e7635bd953215d92a",   # d**k
    "7bc671151cbfaee7f32cd56e86a87b0be30fde8dc72c7f236d3ab2ce42cddbd5",   # p**s
})

# The reverse of obfuscate.py's tricks. "1" stands for both i and l there, so both are tried.
_DIGITS = str.maketrans({"4": "a", "3": "e", "0": "o", "5": "s", "7": "t", "@": "a", "$": "s"})
_HOMOGLYPHS = str.maketrans({"а": "a", "с": "c", "е": "e", "і": "i",
                             "о": "o", "р": "p", "х": "x", "у": "y"})
# Punctuation around a word, except the characters that stand for letters ("@ss", "$hit").
_EDGE = "".join(c for c in string.punctuation if c not in "@$")
_TOKEN = re.compile(r"\S+")


def readings(word: str) -> set[str]:
    """The plain lowercase words ``word`` could be, with the obfuscation tricks undone."""
    core = word.lower().translate(_HOMOGLYPHS).strip(_EDGE)
    plain = core.translate(_DIGITS)
    return {core, plain, plain.replace("1", "i"), plain.replace("1", "l")}


def is_masked(word: str) -> bool:
    return any(hashlib.sha256(r.encode()).hexdigest() in MASKED_TERMS for r in readings(word))


def mask_word(token: str) -> str:
    """``token`` with its inner letters starred if it is a listed word; punctuation kept."""
    if not is_masked(token):
        return token
    start = len(token) - len(token.lstrip(_EDGE))
    end = len(token.rstrip(_EDGE))
    core = token[start:end]
    hidden = core[0] + "*" * (len(core) - 1) if len(core) <= 3 \
        else core[0] + "*" * (len(core) - 2) + core[-1]
    return token[:start] + hidden + token[end:]


def mask(text: str) -> str:
    """``text`` with every listed word masked; spacing and everything else unchanged."""
    return _TOKEN.sub(lambda m: mask_word(m.group(0)), text)

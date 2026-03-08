#!/usr/bin/env python3
"""
Generate 2,500 unique Nepali civic complaints per category (electricity, water, road, garbage).
All text in pure Devanagari. Style mimics Hello Sarkar / OPMCM gunaso portal submissions.
Output: CSV files with columns id,text,category,source,split
Split ratio: 80% train, 10% val, 10% test
"""

import csv
import os
import random
import itertools

random.seed(42)

# ─────────────────────────────────────────────
# SHARED POOLS
# ─────────────────────────────────────────────

LOCATIONS = [
    "काठमाडौं", "ललितपुर", "भक्तपुर", "पोखरा", "वीरगन्ज", "बिराटनगर",
    "जनकपुर", "बुटवल", "हेटौंडा", "धनगढी", "नेपालगञ्ज", "भरतपुर",
    "इटहरी", "दमक", "बिरेन्द्रनगर", "तुलसीपुर", "गुलरिया", "सिद्धार्थनगर",
    "राजविराज", "लहान", "गौर", "कलैया", "त्रिभुवननगर", "कपिलवस्तु",
    "तानसेन", "बागलुङ", "गोरखा", "पाल्पा", "स्याङ्जा", "लमजुङ",
    "कास्की", "म्याग्दी", "मुस्ताङ", "दाङ", "बाँके", "बर्दिया",
    "कैलाली", "कञ्चनपुर", "डोटी", "बैतडी", "दार्चुला", "अछाम",
    "बाजुरा", "हुम्ला", "जुम्ला", "मुगु", "डोल्पा", "रुकुम",
    "सल्यान", "जाजरकोट", "दैलेख", "सुर्खेत", "प्यूठान", "रोल्पा",
    "अर्घाखाँची", "गुल्मी", "रुपन्देही", "नवलपरासी", "चितवन",
    "मकवानपुर", "रामेछाप", "सिन्धुली", "काभ्रेपलाञ्चोक", "सिन्धुपाल्चोक",
    "रसुवा", "नुवाकोट", "धादिङ", "दोलखा", "ओखलढुंगा", "खोटाङ",
    "भोजपुर", "सङ्खुवासभा", "तेह्रथुम", "धनकुटा", "पाँचथर",
    "इलाम", "झापा", "मोरङ", "सुनसरी", "सप्तरी", "सिराहा",
    "महोत्तरी", "धनुषा", "सर्लाही", "रौतहट", "बारा", "पर्सा",
]

WARD_NUMBERS = [str(i) for i in range(1, 33)]

TIMES = [
    "हिजोदेखि", "तीन दिनदेखि", "एक हप्तादेखि", "दुई हप्तादेखि",
    "महिनौँदेखि", "दश दिनदेखि", "पाँच दिनदेखि", "गएको शनिबारदेखि",
    "गत सोमबारदेखि", "बिहीबारदेखि", "चार दिनदेखि", "छ दिनदेखि",
    "लामो समयदेखि", "हप्तौँदेखि", "गएको महिनादेखि", "पन्ध्र दिनदेखि",
    "बीस दिनदेखि", "आज बिहानदेखि", "हिजो रातिदेखि", "गएको शुक्रबारदेखि",
    "दुई तीन दिनदेखि", "केही दिनदेखि", "अस्ति रातिदेखि",
]

EMOTIONS = [
    "अत्यन्त दुःखित छौं", "हामी सबै परेशान छौं", "असह्य भइसकेको छ",
    "कसलाई भन्ने हो थाहा छैन", "अब त सहनै सकिँदैन", "धेरै कष्ट भइरहेको छ",
    "जनतालाई यस्तो सास्ती किन", "यो अन्याय हो", "सरकारले ध्यान दिनुपर्छ",
    "कामै हुँदैन यहाँ", "नागरिकलाई बिर्सिएको जस्तो छ", "न्याय चाहिन्छ",
    "तुरुन्त कारबाही गरिदिनुहोस्", "कहिलेसम्म यस्तै हुन्छ", "अब केही गरिदिनुस्",
    "जिम्मेवार कसले लिन्छ", "यो सेवा होइन, यातना हो", "अब चुप बसिँदैन",
    "प्रशासनले सुन्नुपर्छ", "जनताको गुनासो कसले सुन्छ",
]

CLOSINGS = [
    "तुरुन्त मर्मत गरिदिनुहोस्।", "छिटो समाधान गरिदिनुहोस्।",
    "कृपया ध्यान दिनुहोस्।", "जनतालाई सेवा दिनुहोस्।",
    "हामीलाई कहिलेसम्म कुराउनु पर्ने हो?", "सम्बन्धित निकायले कारबाही गरिदिनुहोस्।",
    "यसमा छानबिन गरिदिनुहोस्।", "जिम्मेवार व्यक्तिलाई कारबाही गरिदिनुहोस्।",
    "कृपया तत्काल कदम चाल्नुहोस्।", "जनताको पीडा बुझ्नुहोस्।",
    "सम्बन्धित वडा कार्यालयबाट कारबाही हुनुपर्छ।", "प्रमुख जिल्ला अधिकारीसम्म पुर्‍याइदिनुहोस्।",
    "हामी थाकिसक्यौं, अब सरकारले हेरोस्।", "यो मानवीय अधिकारको उल्लंघन हो।",
    "कर्मचारीहरूलाई जवाफदेही बनाइदिनुहोस्।",
]


def assign_split(idx, total):
    """80% train, 10% val, 10% test — deterministic by index."""
    r = idx % 10
    if r < 8:
        return "train"
    elif r == 8:
        return "val"
    else:
        return "test"


def pick(pool):
    return random.choice(pool)


def loc_ward():
    return f"{pick(LOCATIONS)} वडा नं {pick(WARD_NUMBERS)}"


def loc_only():
    return pick(LOCATIONS)


# ─────────────────────────────────────────────
# ELECTRICITY
# ─────────────────────────────────────────────

def generate_electricity(n=2500):
    problems = [
        "बत्ती आउँदैन", "बिजुली काटिएको छ", "लाइन गइरहेको छ",
        "बत्ती जाने समस्या निकै बढेको छ", "बिजुली आपूर्ति बन्द छ",
        "बत्ती जानु र आउनु निरन्तर भइरहेको छ",
        "बिजुली नियमित रूपमा काटिन्छ", "बत्ती अफ अन हुने गरेको छ",
    ]
    transformer_issues = [
        "ट्रान्सफर्मर पड्किएको छ", "ट्रान्सफर्मर बिग्रिएको छ",
        "ट्रान्सफर्मरबाट ठूलो आवाज आउँछ", "ट्रान्सफर्मरले लोड धान्न सकेको छैन",
        "ट्रान्सफर्मर फेर्नुपर्ने अवस्था छ", "ट्रान्सफर्मरबाट तेल चुहिएको छ",
        "ट्रान्सफर्मर पुरानो भइसकेको छ", "ट्रान्सफर्मरमा विस्फोट हुने डर छ",
    ]
    voltage_issues = [
        "भोल्टेज धेरै कम आउँछ", "भोल्टेज घटबढ हुन्छ",
        "भोल्टेज अत्यन्त अस्थिर छ", "भोल्टेज फ्लक्चुएसन निकै बढेको छ",
        "भोल्टेज कम हुँदा उपकरण चल्दैनन्", "तीव्र भोल्टेज आउँदा बत्ती पड्किन्छ",
    ]
    wire_issues = [
        "तार झुण्डिएको छ", "तार भुइँमै छुँदैछ", "तार कटिएर खसेको छ",
        "तार पुरानो भएर खुकुलो छ", "बिजुलीको तार रूखमा अल्झिएको छ",
        "तार घरको छानासँग छोइएको छ", "नाङ्गो तार सडकमा झुण्डिएको छ",
    ]
    pole_issues = [
        "पोल बाङ्गो भएको छ", "पोल ढल्ने अवस्थामा छ",
        "पोलबाट चिंगारी निस्किरहेको छ", "पोलबाट स्पार्क आएको छ",
        "पोल सडकतिर झुकेको छ", "पोल पुरानो भएर कुहिएको छ",
    ]
    meter_issues = [
        "मिटर बिग्रिएको छ", "मिटरको रिडिङ गलत देखिन्छ",
        "मिटर जडान अहिलेसम्म भएको छैन", "मिटर अनुमानमा घुम्छ",
        "मिटर बक्स भाँचिएको छ", "मिटरमा देखिएको र बिलमा उल्लेख फरकफरक छ",
    ]
    bill_issues = [
        "बिल अनपेक्षित रूपमा बढी आएको छ", "बिल दोब्बर आएको छ",
        "बिलमा अनावश्यक शुल्क थपिएको छ", "अनुमानित बिल काटिएको छ",
        "बिल तिर्‍यो तर रसिद आएको छैन", "बिलको हिसाब मिल्दैन",
        "बिल अनलाइन तिर्न सकिँदैन", "बिल तिरे पनि लाइन काटिएको छ",
    ]
    streetlight_issues = [
        "स्ट्रीट लाइट बलेको छैन", "सडक बत्ती धेरै दिनदेखि खराब छ",
        "रातमा बाटो अँध्यारो हुन्छ", "स्ट्रीट लाइट जडान नै भएको छैन",
        "सार्वजनिक बत्ती दिनमा बल्छ रातमा बल्दैन",
        "स्ट्रीट लाइट बिग्रेको उजुरी दिए पनि मर्मत भएको छैन",
    ]
    rain_issues = [
        "पानी पर्दा सिधै बत्ती जान्छ", "वर्षामा बिजुली काटिन्छ",
        "हावाहुरी आए पछि बत्ती आउँदैन", "सानो पानी परे पनि लाइन अफ हुन्छ",
        "बर्खामा बत्ती आउनै छोड्छ", "वर्षापछि दिनौँ बत्ती हुँदैन",
    ]
    business_impact = [
        "व्यवसाय ठप्प भएको छ", "पसल चलाउन सकिँदैन",
        "दूध र तरकारी फाल्नुपरेको छ", "होटलमा खाना बनाउन गाह्रो भएको छ",
        "सैलुनको मेसिन चल्दैन", "फ्याक्ट्रीको उत्पादन रोकिएको छ",
        "सुपरमार्केटको फ्रिजर बन्द भयो", "बेकरीको ओभन चलाउन सकिएन",
    ]
    govt_delay = [
        "उजुरी दिएको महिनौँ भयो कुनै कारबाही भएको छैन",
        "कार्यालयमा गएर भन्दा 'हेर्छौं' भन्ने मात्र जवाफ आउँछ",
        "फोन गर्दा कसैले उठाउँदैन",
        "मर्मत टोली आउँछ भनेर भनियो तर अहिलेसम्म आएको छैन",
        "पटकपटक निवेदन दिँदा पनि कुनै नतिजा आएको छैन",
        "वडा कार्यालयले विद्युत् प्राधिकरणलाई पठायो भन्छ, प्राधिकरणले वडालाई भन्छ",
        "जिम्मेवार अधिकारी भेट्नै पाइँदैन",
    ]
    safety_concerns = [
        "बच्चाहरूका लागि ठूलो खतरा छ", "दुर्घटना हुने डर छ",
        "कसैलाई करन्ट लाग्ने खतरा छ", "ज्यान जोखिममा छ",
        "छुँदा सर्ट हुने अवस्था छ", "विद्युत् दुर्घटनाको सम्भावना बढेको छ",
    ]
    impact_phrases = [
        "बालबालिकाको पढाइमा असर परेको छ",
        "स्वास्थ्य सेवामा प्रभाव परेको छ",
        "दैनिक जीवन अस्तव्यस्त भएको छ",
        "कामकाज ठप्प भएको छ",
        "अनलाइन कक्षा सञ्चालन हुन सकेको छैन",
        "फ्रिजमा राखेको खानेकुरा सबै बिग्रिएको छ",
        "पानी तान्ने मोटर चलाउन सकिँदैन",
        "मोबाइल चार्ज गर्नै सकिएको छैन",
        "रातभर अँध्यारोमा बस्नुपरेको छ",
        "बुढापाकालाई निकै गाह्रो भएको छ",
        "गर्भवती महिलालाई कठिनाइ भएको छ",
        "अपाङ्गता भएका व्यक्तिलाई समस्या परेको छ",
    ]

    # Build a large pool of complaint templates (each is a callable)
    templates = []

    # Type 1: Location + time + problem
    def t1():
        return f"{loc_ward()} मा {pick(TIMES)} {pick(problems)}। {pick(impact_phrases)}। {pick(CLOSINGS)}"
    templates.extend([t1] * 12)

    # Type 2: Transformer complaints
    def t2():
        return f"{loc_only()} क्षेत्रमा {pick(transformer_issues)}। {pick(TIMES)} यही समस्या छ। {pick(CLOSINGS)}"
    templates.extend([t2] * 10)

    # Type 3: Voltage issues
    def t3():
        return f"हाम्रो {loc_ward()} मा {pick(voltage_issues)}। {pick(impact_phrases)}। {pick(EMOTIONS)}। {pick(CLOSINGS)}"
    templates.extend([t3] * 10)

    # Type 4: Wire/pole safety
    def t4():
        return f"{loc_only()} मा {pick(wire_issues)}। {pick(safety_concerns)}। {pick(CLOSINGS)}"
    templates.extend([t4] * 8)

    def t4b():
        return f"हाम्रो टोलमा {pick(pole_issues)}। {pick(safety_concerns)}। {pick(EMOTIONS)}।"
    templates.extend([t4b] * 8)

    # Type 5: Meter/bill
    def t5():
        return f"{loc_ward()} मा {pick(meter_issues)}। {pick(bill_issues)}। {pick(CLOSINGS)}"
    templates.extend([t5] * 10)

    def t5b():
        return f"हाम्रो घरको {pick(bill_issues)}। {pick(meter_issues)}। {pick(EMOTIONS)}।"
    templates.extend([t5b] * 8)

    # Type 6: Street light
    def t6():
        return f"{loc_only()} वडा नं {pick(WARD_NUMBERS)} मा {pick(streetlight_issues)}। {pick(safety_concerns)}। {pick(CLOSINGS)}"
    templates.extend([t6] * 8)

    # Type 7: Rain
    def t7():
        return f"{loc_only()} मा {pick(rain_issues)}। {pick(TIMES)} यो समस्या छ। {pick(impact_phrases)}।"
    templates.extend([t7] * 6)

    # Type 8: Business impact
    def t8():
        return f"{loc_ward()} मा बत्ती नआउँदा {pick(business_impact)}। {pick(EMOTIONS)}। {pick(CLOSINGS)}"
    templates.extend([t8] * 8)

    # Type 9: Government delay
    def t9():
        return f"{loc_only()} मा बिजुली सम्बन्धी समस्या छ। {pick(govt_delay)}। {pick(EMOTIONS)}।"
    templates.extend([t9] * 8)

    # Type 10: Compound emotional
    def t10():
        return f"{pick(TIMES)} {loc_ward()} मा {pick(problems)}। {pick(business_impact)}। {pick(impact_phrases)}। {pick(EMOTIONS)}।"
    templates.extend([t10] * 6)

    # Type 11: Direct raw complaint style
    raw_starts = [
        "म {loc} को बासिन्दा हुँ। ",
        "हामी {loc} मा बस्छौं। ",
        "{loc} वडा नं {ward} बाट गुनासो गर्दैछु। ",
        "{loc} का नागरिकको तर्फबाट निवेदन। ",
        "यो गुनासो {loc} वडा नं {ward} बाट हो। ",
        "हेलो सरकार मार्फत {loc} बाट जानकारी गराउँदैछु। ",
        "प्रधानमन्त्री कार्यालयमा {loc} बाट निवेदन। ",
    ]

    def t11():
        start = pick(raw_starts).format(loc=loc_only(), ward=pick(WARD_NUMBERS))
        body = pick([
            f"{pick(problems)}। {pick(impact_phrases)}।",
            f"{pick(transformer_issues)}, {pick(problems)}।",
            f"{pick(voltage_issues)}, {pick(impact_phrases)}।",
            f"{pick(wire_issues)}, {pick(safety_concerns)}।",
            f"{pick(meter_issues)}, {pick(bill_issues)}।",
            f"{pick(streetlight_issues)}, {pick(safety_concerns)}।",
            f"{pick(rain_issues)}, {pick(TIMES)} यही हालत छ।",
        ])
        return f"{start}{body} {pick(CLOSINGS)}"
    templates.extend([t11] * 15)

    # Type 12: Short frustrated
    def t12():
        return f"{loc_only()} मा {pick(TIMES)} {pick(problems)}, {pick(EMOTIONS)}!"
    templates.extend([t12] * 8)

    # Type 13: Comparison / inequality
    def t13():
        l1, l2 = random.sample(LOCATIONS, 2)
        return f"{l1} मा बत्ती ठिक छ तर नजिकैको {l2} मा {pick(problems)}। {pick(EMOTIONS)}। किन यस्तो भेदभाव?"
    templates.extend([t13] * 5)

    # Type 14: Multiple issues combined
    def t14():
        return f"{loc_ward()} मा {pick(problems)}, {pick(transformer_issues)}, र {pick(voltage_issues)}। {pick(EMOTIONS)}। {pick(CLOSINGS)}"
    templates.extend([t14] * 8)

    # Type 15: Question style
    def t15():
        return f"{loc_only()} मा {pick(TIMES)} {pick(problems)}। कहिले आउँछ बत्ती? {pick(EMOTIONS)}।"
    templates.extend([t15] * 5)

    # Type 16: Time-specific
    time_specific = [
        "बिहान ६ बजेदेखि", "साँझ ५ बजेदेखि", "राति ८ बजेदेखि",
        "दिउँसो १ बजेदेखि", "राति ११ बजेदेखि", "बिहान ४ बजेदेखि",
        "साँझ ७ बजेपछि", "दिउँसो ३ बजेतिर",
    ]
    def t16():
        return f"आज {pick(time_specific)} {loc_only()} मा {pick(problems)}। {pick(impact_phrases)}। {pick(CLOSINGS)}"
    templates.extend([t16] * 6)

    # Type 17: Family impact
    family = [
        "हाम्रो परिवारमा सानो बच्चा छ", "घरमा बुढाबुढी मात्र बस्नुहुन्छ",
        "गर्भवती श्रीमती घरमा हुनुहुन्छ", "बिरामी बुवा घरमा हुनुहुन्छ",
        "परीक्षा दिने छोराछोरी छन्", "नवजात शिशु भएको परिवार हो",
    ]
    def t17():
        return f"{pick(family)}। {loc_ward()} मा {pick(TIMES)} {pick(problems)}। {pick(impact_phrases)}। {pick(CLOSINGS)}"
    templates.extend([t17] * 6)

    # Type 18: Follow-up / repeated complaint
    def t18():
        return f"यो दोस्रो / तेस्रो पटकको गुनासो हो। {loc_only()} मा {pick(problems)}। {pick(govt_delay)}। {pick(EMOTIONS)}।"
    templates.extend([t18] * 5)

    # Generate unique complaints
    seen = set()
    rows = []
    attempts = 0
    max_attempts = n * 20

    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        t = pick(templates)()
        # Normalize whitespace
        t = " ".join(t.split())
        if t not in seen:
            seen.add(t)
            rows.append(t)

    return rows


# ─────────────────────────────────────────────
# WATER
# ─────────────────────────────────────────────

def generate_water(n=2500):
    no_water = [
        "धारामा पानी आएको छैन", "खानेपानी आपूर्ति बन्द छ",
        "पानी आउनै छोडेको छ", "धारा सुक्खा छ",
        "पानीको एक थोपा पनि आएको छैन", "धारामा हावा मात्र आउँछ",
        "बाल्टी राखेको ठाउँमा पानी नै छैन", "लाइनमा पानी नै हुँदैन",
    ]
    dirty_water = [
        "फोहोर पानी आउँछ", "पहेँलो रङको पानी आएको छ",
        "दुर्गन्धित पानी आउँछ", "पानीमा माटो मिसिएको छ",
        "हरियो रङको पानी आउँछ", "पानी गन्हाउँछ",
        "कीरा मिसिएको पानी आउँछ", "पानीमा तेलो पदार्थ देखिन्छ",
    ]
    pipe_issues = [
        "पाइप फुटेको छ", "पाइपबाट पानी चुहिरहेको छ",
        "पाइपलाइन पुरानो भइसकेको छ", "जोड्ने ठाउँबाट पानी बगेको छ",
        "मुख्य पाइप क्षतिग्रस्त भएको छ", "पाइप भाँचिएर सडकमा पानी बगेको छ",
    ]
    pressure_issues = [
        "पानीको प्रेसर निकै कम छ", "धारामा थोपा थोपा मात्र आउँछ",
        "माथिल्लो तल्लामा पानी चढ्दैन", "ट्यांकी भर्न घण्टौं लाग्छ",
        "प्रेसर कम भएर नुहाउन पनि सकिँदैन",
    ]
    timing_issues = [
        "पानी आउने समय अनियमित छ", "राति मात्र पानी आउँछ",
        "बिहान ४ बजे मात्र पानी आउँछ", "तोकिएको समयमा पानी आउँदैन",
        "हप्तामा एकपटक मात्र पानी दिइन्छ", "पानीको तालिका पालना हुँदैन",
    ]
    tanker_issues = [
        "ट्यांकर धेरै ढिलो आउँछ", "ट्यांकर आउने भनेर आएको छैन",
        "ट्यांकरको पानी पनि फोहोर हुन्छ", "ट्यांकरको भाडा निकै महँगो छ",
        "ट्यांकर पालैपालो दिँदा झगडा हुन्छ",
    ]
    summer_issues = [
        "गर्मीमा पानीको चरम अभाव छ", "गर्मी लाग्दा पानी हराउँछ",
        "जेठ असारमा धारा सुक्खा हुन्छ", "गर्मीमा ट्यांकी खाली नै रहन्छ",
    ]
    health_concern = [
        "यो पानी पिउँदा ढाड दुख्छ", "फोहोर पानीले छालारोग भएको छ",
        "बच्चालाई झाडापखाला भएको छ", "यो पानी पिएर बिरामी परिने डर छ",
        "पानीजन्य रोग फैलिने खतरा छ", "पानी उमाल्दा पनि सफा हुँदैन",
    ]
    well_source = [
        "इनार सुकिसकेको छ", "ट्युबवेलमा पानी आउँदैन",
        "कुवाको पानी पनि तल गइसकेको छ", "बोरिङ गरे पनि पानी भेटिएन",
    ]
    impact_w = [
        "खाना पकाउन सकिँदैन", "भाँडा माझ्न पानी छैन",
        "कपडा धुन पानी छैन", "शौचालय प्रयोग गर्न अप्ठ्यारो छ",
        "जनावरलाई पनि पानी दिन गाह्रो छ", "बच्चालाई नुहाउन पानी छैन",
        "बिरामीलाई कसरी हेर्ने पानी बिनै", "दैनिक कामकाज ठप्प छ",
    ]

    templates = []

    def w1():
        return f"{loc_ward()} मा {pick(TIMES)} {pick(no_water)}। {pick(impact_w)}। {pick(CLOSINGS)}"
    templates.extend([w1] * 12)

    def w2():
        return f"{loc_only()} मा {pick(dirty_water)}। {pick(health_concern)}। {pick(CLOSINGS)}"
    templates.extend([w2] * 10)

    def w3():
        return f"हाम्रो {loc_ward()} मा {pick(pipe_issues)}। {pick(TIMES)} यही अवस्था छ। {pick(CLOSINGS)}"
    templates.extend([w3] * 8)

    def w4():
        return f"{loc_only()} क्षेत्रमा {pick(pressure_issues)}। {pick(impact_w)}। {pick(EMOTIONS)}।"
    templates.extend([w4] * 8)

    def w5():
        return f"{loc_ward()} मा {pick(timing_issues)}। {pick(impact_w)}। {pick(CLOSINGS)}"
    templates.extend([w5] * 8)

    def w6():
        return f"{loc_only()} मा {pick(tanker_issues)}। {pick(EMOTIONS)}। {pick(CLOSINGS)}"
    templates.extend([w6] * 6)

    def w7():
        return f"{pick(summer_issues)}। {loc_ward()} मा {pick(no_water)}। {pick(EMOTIONS)}।"
    templates.extend([w7] * 6)

    def w8():
        return f"म {loc_only()} को बासिन्दा हुँ। {pick(TIMES)} {pick(no_water)}। {pick(impact_w)}। {pick(CLOSINGS)}"
    templates.extend([w8] * 10)

    def w9():
        return f"{loc_only()} वडा नं {pick(WARD_NUMBERS)} बाट गुनासो। {pick(dirty_water)}, {pick(health_concern)}। {pick(CLOSINGS)}"
    templates.extend([w9] * 8)

    def w10():
        return f"हाम्रो टोलमा {pick(well_source)}। {pick(no_water)}। {pick(EMOTIONS)}।"
    templates.extend([w10] * 5)

    def w11():
        return f"{loc_ward()} मा {pick(pipe_issues)}, {pick(no_water)}। {pick(TIMES)} यो समस्या जारी छ। {pick(CLOSINGS)}"
    templates.extend([w11] * 6)

    def w12():
        return f"हेलो सरकार मार्फत गुनासो। {loc_only()} मा {pick(no_water)}। {pick(timing_issues)}। {pick(impact_w)}। {pick(CLOSINGS)}"
    templates.extend([w12] * 6)

    def w13():
        return f"{loc_only()} मा {pick(TIMES)} {pick(no_water)}, {pick(EMOTIONS)}!"
    templates.extend([w13] * 5)

    def w14():
        return f"पटकपटक गुनासो गर्दा पनि {loc_only()} मा पानी आएको छैन। {pick(EMOTIONS)}। {pick(CLOSINGS)}"
    templates.extend([w14] * 5)

    def w15():
        return f"घरमा {pick(['सानो बच्चा छ', 'बुढाबुढी हुनुहुन्छ', 'बिरामी हुनुहुन्छ', 'गर्भवती श्रीमती हुनुहुन्छ'])}। {loc_ward()} मा {pick(no_water)}। {pick(impact_w)}। {pick(CLOSINGS)}"
    templates.extend([w15] * 6)

    seen = set()
    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 20:
        attempts += 1
        t = pick(templates)()
        t = " ".join(t.split())
        if t not in seen:
            seen.add(t)
            rows.append(t)
    return rows


# ─────────────────────────────────────────────
# ROAD
# ─────────────────────────────────────────────

def generate_road(n=2500):
    potholes = [
        "सडकमा ठूला खाल्डा परेका छन्", "सडकभरि खाल्डाखुल्डी छ",
        "खाल्डाले सवारी चलाउन गाह्रो भएको छ", "सडकमा गहिरा खाडल छन्",
        "खाल्डा भित्र पानी भरिएको छ", "सडक खाल्डाखुल्डीले भरिएको छ",
    ]
    road_damage = [
        "सडक पूरै भत्किएको छ", "सडकको कालोपत्रे उप्किएको छ",
        "सडक टुक्रा टुक्रा भएको छ", "ग्राभेल सडक भासिएको छ",
        "सडकको सतह बिग्रिएको छ", "कालोपत्रे सडक चिर्‍याएको जस्तो छ",
    ]
    muddy_road = [
        "वर्षामा सडक हिलोमुलो हुन्छ", "सडकमा हिलो मात्र छ",
        "पानी परेपछि सडक दलदल हुन्छ", "हिलामा पुरिने सडक छ",
        "कच्ची सडक वर्षातमा हिँड्नै नसक्ने हुन्छ",
    ]
    bridge_issues = [
        "पुल भत्किएको छ", "पुलमा दरार देखिएको छ",
        "साँघुरो पुल धेरै जोखिमपूर्ण छ", "खोलामा पुल नै छैन",
        "पुरानो काठे पुल अहिले पनि प्रयोगमा छ",
    ]
    no_repair = [
        "वर्षौँदेखि मर्मत भएको छैन", "ठेकेदारले काम अधुरो छोडेको छ",
        "सडक बनाउने भनेर बजेट आयो तर काम भएन",
        "बजेट विनियोजन भएको छ तर सडक उस्तै छ",
        "गएको वर्ष बनाइएको सडक एकै वर्षमा भत्कियो",
    ]
    accident_risk = [
        "दुर्घटनाको जोखिम बढेको छ", "मोटरसाइकल पटकपटक दुर्घटना हुन्छ",
        "स्कुल बसलाई पनि गाह्रो छ", "एम्बुलेन्स समयमा पुग्न सक्दैन",
        "बालबालिका स्कुल जान डराउँछन्",
    ]
    blocked_road = [
        "सडक अवरुद्ध छ", "भूस्खलनले बाटो बन्द भएको छ",
        "सडकमा रुख ढलेको छ", "निर्माण सामग्रीले बाटो छोपेको छ",
        "नाली नखन्दा पानी सडकमा भरिन्छ",
    ]
    drainage = [
        "नालीको पानी सडकमा बग्छ", "ढलको पानी सडकमा भरिन्छ",
        "नाली बन्द भएर सडक डुब्छ", "सडकमा पानी जम्मा हुन्छ",
    ]
    dust = [
        "धुलो उड्दा सास फेर्न गाह्रो हुन्छ", "ग्राभेल सडकको धुलो असह्य छ",
        "धुलोले टोलका बिरामी बढेका छन्", "गर्मीमा सडकको धुलो नसहने भएको छ",
    ]
    footpath = [
        "फुटपाथ भत्किएको छ", "फुटपाथ बनाइएको छैन",
        "पैदल यात्रुका लागि ठाउँ छैन", "फुटपाथमा पसल राखिएको छ",
    ]
    impact_r = [
        "सवारी चलाउन अत्यन्त गाह्रो छ", "हिँड्न सकिने अवस्था छैन",
        "ज्येष्ठ नागरिकलाई कठिनाइ भएको छ", "विद्यालय जाने बच्चालाई गाह्रो छ",
        "बिरामीलाई अस्पताल पु‍र्‍याउन गाह्रो", "ढुवानी सेवा प्रभावित भएको छ",
    ]
    landslide = [
        "पहिरो गएर बाटो बन्द छ", "माटो खसेर सडक पुरिएको छ",
        "पहिरोले सडक भत्काएको छ", "चट्टान खसेर बाटो अवरुद्ध छ",
    ]

    templates = []

    def r1():
        return f"{loc_ward()} मा {pick(potholes)}। {pick(impact_r)}। {pick(CLOSINGS)}"
    templates.extend([r1] * 12)

    def r2():
        return f"{loc_only()} क्षेत्रमा {pick(road_damage)}। {pick(TIMES)} यो हालत छ। {pick(CLOSINGS)}"
    templates.extend([r2] * 10)

    def r3():
        return f"हाम्रो {loc_ward()} मा {pick(muddy_road)}। {pick(impact_r)}। {pick(EMOTIONS)}।"
    templates.extend([r3] * 8)

    def r4():
        return f"{loc_only()} मा {pick(bridge_issues)}। {pick(accident_risk)}। {pick(CLOSINGS)}"
    templates.extend([r4] * 6)

    def r5():
        return f"{loc_ward()} मा {pick(road_damage)}। {pick(no_repair)}। {pick(EMOTIONS)}।"
    templates.extend([r5] * 8)

    def r6():
        return f"{loc_only()} मा {pick(blocked_road)}। {pick(impact_r)}। {pick(CLOSINGS)}"
    templates.extend([r6] * 6)

    def r7():
        return f"{loc_ward()} मा {pick(drainage)}। {pick(muddy_road)}। {pick(CLOSINGS)}"
    templates.extend([r7] * 6)

    def r8():
        return f"म {loc_only()} को बासिन्दा हुँ। {pick(potholes)}। {pick(accident_risk)}। {pick(CLOSINGS)}"
    templates.extend([r8] * 8)

    def r9():
        return f"हेलो सरकार मार्फत गुनासो। {loc_only()} मा {pick(road_damage)}, {pick(no_repair)}। {pick(EMOTIONS)}।"
    templates.extend([r9] * 6)

    def r10():
        return f"{loc_only()} मा {pick(dust)}। {pick(TIMES)} यो समस्या छ। {pick(CLOSINGS)}"
    templates.extend([r10] * 5)

    def r11():
        return f"{loc_ward()} मा {pick(footpath)}। {pick(impact_r)}। {pick(EMOTIONS)}।"
    templates.extend([r11] * 5)

    def r12():
        return f"{loc_only()} मा {pick(landslide)}। {pick(TIMES)} बाटो बन्द छ। {pick(impact_r)}। {pick(CLOSINGS)}"
    templates.extend([r12] * 5)

    def r13():
        return f"{loc_ward()} मा {pick(potholes)}, {pick(road_damage)}। {pick(accident_risk)}। {pick(EMOTIONS)}। {pick(CLOSINGS)}"
    templates.extend([r13] * 6)

    def r14():
        return f"पटकपटक निवेदन दिँदा पनि {loc_only()} मा सडक मर्मत भएको छैन। {pick(EMOTIONS)}। {pick(CLOSINGS)}"
    templates.extend([r14] * 5)

    def r15():
        return f"आज {pick(TIMES)} {loc_only()} मा {pick(potholes)}, {pick(EMOTIONS)}!"
    templates.extend([r15] * 4)

    seen = set()
    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 20:
        attempts += 1
        t = pick(templates)()
        t = " ".join(t.split())
        if t not in seen:
            seen.add(t)
            rows.append(t)
    return rows


# ─────────────────────────────────────────────
# GARBAGE
# ─────────────────────────────────────────────

def generate_garbage(n=2500):
    not_collected = [
        "फोहोर उठाइएको छैन", "फोहोर उठ्न आएको छैन",
        "हप्तौँदेखि फोहोर उठ्ने गाडी आएको छैन",
        "फोहोर सङ्कलन पूर्ण रूपमा बन्द छ",
        "नगरपालिकाले फोहोर उठाउन छोडेको छ",
        "फोहोर सङ्कलन गर्ने गाडी आउँदैन",
    ]
    overflowing = [
        "डस्टबिन भरिएर बग्दैछ", "कचरा पेटी ओभरफ्लो भएको छ",
        "डस्टबिन वरिपरि फोहोर छरिएको छ",
        "सडक छेउको डस्टबिन भरिँदा बाटोमै छरिएको छ",
    ]
    smell = [
        "असह्य दुर्गन्ध फैलिएको छ", "नाक थुनेर हिँड्नुपर्ने अवस्था छ",
        "गन्धले बस्नै गाह्रो भएको छ", "कुहिएको फोहोरको गन्ध सहन सकिँदैन",
        "टोलभरि दुर्गन्ध छ", "बिहानदेखि राति सम्म गन्ध आइरहेको छ",
    ]
    road_garbage = [
        "सडकमा फोहोर थुपारिएको छ", "बाटोमा कचरा छरिएको छ",
        "सडकको बीचमै फोहोरको थुप्रो छ", "फुटपाथमा फोहोर जम्मा भएको छ",
    ]
    drain_block = [
        "फोहोरले नाली बन्द भएको छ", "ढलमा फोहोर थुनिएको छ",
        "कचराले ढल अवरुद्ध भएर पानी सडकमा बग्छ",
        "नाली थुनिँदा पानी घरभित्रै पस्छ",
    ]
    animals = [
        "कुकुरले फोहोर छर्‍याउँछ", "गाईले फोहोर फिँजाउँछ",
        "मुसाले फोहोर कोट्याउँछ", "बाँदरले कचरा फैलाउँछ",
        "स्वेत कुकुरको बथानले फोहोर छरिदिन्छ",
    ]
    public_places = [
        "विद्यालय नजिक फोहोर थुपारिएको छ",
        "अस्पताल अगाडि कचरा जम्मा भएको छ",
        "मन्दिर वरिपरि फोहोर छ", "पार्कमा कचरा छ",
        "बसपार्कमा फोहोरमय छ", "चोकमा फोहोरको डंगुर छ",
    ]
    health_g = [
        "रोग फैलिने डर छ", "बच्चालाई बिरामी पार्ने अवस्था छ",
        "मक्खी र लामखुट्टे बढेका छन्", "छालारोग र श्वासप्रश्वासको समस्या बढेको छ",
        "फोहोरबाट सार्वजनिक स्वास्थ्य खतरामा छ",
    ]
    burning = [
        "मान्छेले फोहोर जलाउँदा धुवाँ फैलिन्छ",
        "फोहोर पोल्दा सास फेर्न गाह्रो हुन्छ",
        "रातमा कसैले फोहोर बालिदिन्छ",
    ]
    truck_delay = [
        "फोहोर उठाउने गाडी समयमा आउँदैन",
        "गाडी हप्ताको एक पटक मात्र आउँछ",
        "फोहोर उठाउने गाडीको सेवा अनियमित छ",
        "गाडी आउने तालिका पालना हुँदैन",
    ]
    segregation = [
        "फोहोर वर्गीकरण गरे पनि मिश्रित लगिन्छ",
        "सुख्खा ओसिलो छुट्याए पनि एकैमा हालिन्छ",
        "छुट्याएर राखे पनि गाडीले सबै मिसाउँछ",
    ]

    templates = []

    def g1():
        return f"{loc_ward()} मा {pick(TIMES)} {pick(not_collected)}। {pick(smell)}। {pick(CLOSINGS)}"
    templates.extend([g1] * 12)

    def g2():
        return f"{loc_only()} क्षेत्रमा {pick(overflowing)}। {pick(road_garbage)}। {pick(CLOSINGS)}"
    templates.extend([g2] * 8)

    def g3():
        return f"हाम्रो {loc_ward()} मा {pick(smell)}। {pick(health_g)}। {pick(EMOTIONS)}।"
    templates.extend([g3] * 8)

    def g4():
        return f"{loc_only()} मा {pick(drain_block)}। {pick(TIMES)} यही अवस्था छ। {pick(CLOSINGS)}"
    templates.extend([g4] * 6)

    def g5():
        return f"{loc_ward()} मा {pick(animals)}। {pick(road_garbage)}। {pick(EMOTIONS)}।"
    templates.extend([g5] * 6)

    def g6():
        return f"{loc_only()} मा {pick(public_places)}। {pick(health_g)}। {pick(CLOSINGS)}"
    templates.extend([g6] * 6)

    def g7():
        return f"म {loc_only()} को बासिन्दा हुँ। {pick(not_collected)}। {pick(smell)}। {pick(CLOSINGS)}"
    templates.extend([g7] * 8)

    def g8():
        return f"हेलो सरकार मार्फत गुनासो। {loc_only()} मा {pick(not_collected)}, {pick(smell)}। {pick(EMOTIONS)}।"
    templates.extend([g8] * 6)

    def g9():
        return f"{loc_ward()} मा {pick(burning)}। {pick(health_g)}। {pick(CLOSINGS)}"
    templates.extend([g9] * 5)

    def g10():
        return f"{loc_only()} मा {pick(truck_delay)}। {pick(not_collected)}। {pick(EMOTIONS)}। {pick(CLOSINGS)}"
    templates.extend([g10] * 6)

    def g11():
        return f"{loc_ward()} मा {pick(segregation)}। {pick(EMOTIONS)}। {pick(CLOSINGS)}"
    templates.extend([g11] * 5)

    def g12():
        return f"पटकपटक गुनासो गर्दा पनि {loc_only()} मा फोहोर उठ्ने गरेको छैन। {pick(EMOTIONS)}। {pick(CLOSINGS)}"
    templates.extend([g12] * 5)

    def g13():
        return f"{loc_only()} मा {pick(TIMES)} {pick(not_collected)}, {pick(EMOTIONS)}!"
    templates.extend([g13] * 5)

    def g14():
        return f"घरमा {pick(['सानो बच्चा छ', 'बुढाबुढी हुनुहुन्छ', 'बिरामी हुनुहुन्छ'])}। {loc_ward()} मा {pick(smell)}। {pick(health_g)}। {pick(CLOSINGS)}"
    templates.extend([g14] * 5)

    def g15():
        return f"{loc_ward()} मा {pick(not_collected)}, {pick(overflowing)}, र {pick(smell)}। {pick(EMOTIONS)}। {pick(CLOSINGS)}"
    templates.extend([g15] * 6)

    seen = set()
    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 20:
        attempts += 1
        t = pick(templates)()
        t = " ".join(t.split())
        if t not in seen:
            seen.add(t)
            rows.append(t)
    return rows


# ─────────────────────────────────────────────
# MAIN: write all four CSVs
# ─────────────────────────────────────────────

def write_csv(filepath, category, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "text", "category", "source", "split"])
        for idx, text in enumerate(rows):
            row_id = idx + 1
            split = assign_split(idx, len(rows))
            writer.writerow([row_id, text, category, "hello_sarkar", split])
    print(f"✓ {filepath}: {len(rows)} complaints written")


def main():
    base = os.path.join(os.path.dirname(__file__), "data")

    print("Generating electricity complaints...")
    elec = generate_electricity(2500)
    write_csv(os.path.join(base, "electricity_nepali.csv"), "electricity", elec)

    print("Generating water complaints...")
    water = generate_water(2500)
    write_csv(os.path.join(base, "water_nepali.csv"), "water", water)

    print("Generating road complaints...")
    road = generate_road(2500)
    write_csv(os.path.join(base, "road_nepali.csv"), "road", road)

    print("Generating garbage complaints...")
    garb = generate_garbage(2500)
    write_csv(os.path.join(base, "garbage_nepali.csv"), "garbage", garb)

    print(f"\nTotal: {len(elec) + len(water) + len(road) + len(garb)} complaints generated.")


if __name__ == "__main__":
    main()

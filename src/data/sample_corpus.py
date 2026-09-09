"""Deterministic Apple Support–style sample matching the Kaggle tweet schema.

This is NOT a dump of the Kaggle corpus. It exists so the repository runs
without Kaggle credentials and without committing 3M tweets. Brand selection
and the intent taxonomy are still grounded in published analyses of the real
dataset (see docs/brand_selection.md and docs/intent_taxonomy.md).

Schema (thoughtvector/customer-support-on-twitter):
    tweet_id, author_id, inbound, created_at, text, response_tweet_id,
    in_response_to_tweet_id
"""

from __future__ import annotations

import calendar
import random
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.config import intent_names
from src.paths import SAMPLE, ensure_dirs

BRAND = "AppleSupport"

DEVICES = [
    "iPhone 6",
    "iPhone 6s",
    "iPhone 7",
    "iPhone 7 Plus",
    "iPhone 8",
    "iPhone 8 Plus",
    "iPhone X",
    "iPad Pro",
    "iPad",
    "Apple Watch",
    "MacBook Pro",
]
IOS = ["10.3.3", "11.0", "11.0.1", "11.0.2", "11.1", "11.2"]
APPS = ["Safari", "Mail", "Messages", "Photos", "Music", "Notes", "Camera"]
EXTRAS = [
    "Please help.",
    "This is so frustrating.",
    "Any advice?",
    "It's been two days.",
    "I already tried restarting.",
    "pls fix",
    "",
]


def _created(rng: random.Random) -> str:
    start = datetime(2017, 10, 1, tzinfo=timezone.utc)
    stamp = start + timedelta(minutes=rng.randint(0, 60 * 24 * 50))
    wday = calendar.day_abbr[stamp.weekday()]
    mon = calendar.month_abbr[stamp.month]
    return f"{wday} {mon} {stamp.day:02d} {stamp:%H:%M:%S} +0000 {stamp.year}"


def _user(rng: random.Random) -> str:
    return rng.choice(
        [
            "frustrated_parent",
            "ios_user",
            "jane_c",
            "dev_mike",
            "samk",
            "watchfan",
            "photo_hobby",
            "college_rob",
            "maria_t",
            "nyc_tom",
            "uk_ellen",
            "pixel_skeptic",
            "family_plan",
            "smallbiz_ann",
        ]
    ) + str(rng.randint(10, 99))


CUSTOMERS: dict[str, list[str]] = {
    "ios_update_bug": [
        "@AppleSupport my {device} keeps {symptom} after I updated to iOS {ios}. {extra}",
        "@AppleSupport iOS {ios} made my {device} unusable. {symptom}. {extra}",
        "@AppleSupport ever since the iOS {ios} update {symptom} on my {device}",
        "@AppleSupport why is iOS {ios} so buggy on {device}? {app} {symptom}",
        "@AppleSupport {app} keeps crashing on iOS {ios} {device} {extra}",
        "@AppleSupport I installed iOS {ios} last night and now my {device} {symptom}",
        "@AppleSupport is anyone else's {device} {symptom} after the iOS {ios} update",
        "@AppleSupport can I roll back from iOS {ios}? {device} {symptom} constantly",
        "@AppleSupport iOS {ios} update failed and my {device} is stuck on the Apple logo",
        "@AppleSupport after updating {device} to {ios}, {app} closes as soon as I open it",
        "@AppleSupport the latest iOS update bricked my {device}. looping on boot. help",
        "@AppleSupport Mail and Safari freeze on iOS {ios}. {device}. this started after the update",
    ],
    "hardware_device": [
        "@AppleSupport {device} battery drains to 0% in {hours} hours with almost no use",
        "@AppleSupport my {device} won't charge. tried 3 cables and 2 bricks. {extra}",
        "@AppleSupport screen on my {device} has {screen}. is this a hardware defect?",
        "@AppleSupport {device} gets burning hot even when I'm just texting. {extra}",
        "@AppleSupport my {device} will not turn on at all. apple logo then black",
        "@AppleSupport Touch ID / home button on {device} stopped working this morning",
        "@AppleSupport speaker on {device} sounds crackly and the mic cuts out on calls",
        "@AppleSupport dropped {device} from waist height and now {screen}. covered?",
        "@AppleSupport battery health already {pct}% after {months} months on {device}",
        "@AppleSupport {device} randomly reboots during the day. not after an update, been on this ios for months",
        "@AppleSupport charging port on {device} is loose. only charges at a certain angle",
        "@AppleSupport my {device} screen is unresponsive in the bottom third. hardware?",
    ],
    "apple_id_icloud": [
        "@AppleSupport my Apple ID is locked and the reset email never arrives. {extra}",
        "@AppleSupport I cannot sign in to iCloud on my new {device}. password is correct",
        "@AppleSupport two-factor is sending codes to a phone I no longer have. locked out",
        "@AppleSupport keep getting Apple ID verification popups I cannot dismiss on {device}",
        "@AppleSupport forgot Apple ID password and the security questions are from 2012 help",
        "@AppleSupport someone is trying to access my Apple ID. I got emails about it. what do I do",
        "@AppleSupport iCloud sign-in loop on {device} after restore. it just shakes and fails",
        "@AppleSupport Apple ID says locked for security reasons. I did not change anything",
        "@AppleSupport can you unlock my Apple ID? I need my photos for work tomorrow",
        "@AppleSupport iCloud keychain and Apple ID keep asking me to sign in every 10 minutes",
        "@AppleSupport I never set up two-factor but now I'm blocked from my Apple ID",
        "@AppleSupport shared family Apple ID is locked and my kids cannot download apps",
    ],
    "billing_subscription": [
        "@AppleSupport I was charged twice for Apple Music this month. {extra}",
        "@AppleSupport there is a charge I did not authorize on my App Store account",
        "@AppleSupport I cancelled Apple Music last month but you still billed me",
        "@AppleSupport how do I get a refund for an app my kid bought by accident",
        "@AppleSupport iCloud storage billed me after I turned it off. please refund",
        "@AppleSupport duplicate iTunes charge on my card today. I only bought one album",
        "@AppleSupport subscription I never signed up for is renewing. billed again",
        "@AppleSupport charged for an in-app purchase that failed to deliver. want a refund",
        "@AppleSupport Apple Music family plan billed my personal card not the organizer",
        "@AppleSupport why was I charged for iCloud 50GB when I am still on 5GB free",
        "@AppleSupport my receipt shows two App Store purchases I did not make",
        "@AppleSupport I requested a refund 5 days ago and have heard nothing. still charged",
    ],
    "connectivity": [
        "@AppleSupport {device} will not stay connected to Wi-Fi. drops every few minutes",
        "@AppleSupport Bluetooth keeps dropping my AirPods on {device}. {extra}",
        "@AppleSupport CarPlay disconnects as soon as I start driving. iOS {ios}",
        "@AppleSupport cellular data barely works indoors on {device} after nothing changed",
        "@AppleSupport AirDrop not seeing my Mac from {device}. both have it on",
        "@AppleSupport Wi-Fi greyed out on {device}. cannot toggle it at all",
        "@AppleSupport hotspot from {device} connects then loses internet immediately",
        "@AppleSupport {device} forgets known Wi-Fi networks every reboot",
        "@AppleSupport LTE is stuck on 3G / no service with a full SIM. {device}",
        "@AppleSupport cannot pair Apple Watch to {device}. Bluetooth dance fails",
        "@AppleSupport Wi-Fi calling dropped 4 times today on {device}",
        "@AppleSupport personal hotspot drains and disconnects when my laptop joins",
    ],
    "backup_storage": [
        "@AppleSupport iCloud says I am out of storage but I only have a few hundred photos",
        "@AppleSupport I restored my {device} and all my photos are gone. {extra}",
        "@AppleSupport iCloud backup failed every night this week on {device}",
        "@AppleSupport Messages disappeared after restore. can they come back from iCloud",
        "@AppleSupport not enough iCloud storage to backup and I already deleted videos",
        "@AppleSupport photos not uploading to iCloud from {device} for 3 days",
        "@AppleSupport I paid for 200GB but backup still says storage full",
        "@AppleSupport how do I recover contacts I lost when I set up {device} as new",
        "@AppleSupport iCloud backup is 2 months old and won't refresh. {device}",
        "@AppleSupport I turned on iCloud Photos and now originals vanished from {device}",
        "@AppleSupport restore from iCloud is stuck at {pct}% for hours",
        "@AppleSupport I need last week's backup, not the one from this morning that overwrote it",
    ],
    "order_purchase": [
        "@AppleSupport my {item} order still says processing after {days} days. {extra}",
        "@AppleSupport the {item} I received has a damaged box and {damage}",
        "@AppleSupport pickup email never arrived for my Apple Store reservation",
        "@AppleSupport tracking number for my {item} has not updated in a week",
        "@AppleSupport I was charged for {item} but the order was cancelled on the site",
        "@AppleSupport delivery said left at door. nothing there. {item} order",
        "@AppleSupport can I change the shipping address on my {item} order? not shipped yet",
        "@AppleSupport Apple Store said my {item} is ready but they cannot find it",
        "@AppleSupport preorder {item} slipped from launch week to sometime next month",
        "@AppleSupport missing AirPods / charger in the {item} box I just opened",
        "@AppleSupport I need to cancel my {item} order before it ships. site errors",
        "@AppleSupport two {item} charges, one order. shipping status is blank",
    ],
    "how_to_feature": [
        "@AppleSupport how do I turn off Wi-Fi Assist on iOS {ios}?",
        "@AppleSupport where is Do Not Disturb on {device}?",
        "@AppleSupport does {device} support wireless charging?",
        "@AppleSupport how to take a screenshot on {device}",
        "@AppleSupport how do I stop {app} from using cellular data",
        "@AppleSupport where can I see battery usage by app on iOS {ios}",
        "@AppleSupport is there a way to hide photos without deleting them",
        "@AppleSupport how do I set up Do Not Disturb while driving",
        "@AppleSupport what does the little moon icon mean in the status bar",
        "@AppleSupport can I use two WhatsApp accounts on {device}",
        "@AppleSupport how to move apps back to the dock on iOS {ios}",
        "@AppleSupport does iOS {ios} let me record the screen without a Mac",
    ],
    "other": [
        "@AppleSupport you guys are the worst. that is all.",
        "@AppleSupport why is the store in my mall so crowded on Saturday",
        "hey @AppleSupport just saying hi to Tim Cook",
        "@AppleSupport my nephew's school blocks iMessage. fix the school",
        "@AppleSupport your commercials are better than the phones",
        "@AppleSupport can you tell Verizon to give me a better plan",
        "@AppleSupport I lost a bet and have to tweet you. nothing is wrong",
        "@AppleSupport love the packaging. that is the tweet",
        "@AppleSupport is it true you are launching a car. asking for a friend",
        "@AppleSupport the person ahead of me at Genius Bar is taking forever",
        "@AppleSupport please follow me back",
        "@AppleSupport this is a tweet about Android being cheaper. thoughts?",
    ],
}

SYMPTOMS = [
    "restarting",
    "crashing",
    "freezing",
    "rebooting",
    "getting stuck",
    "apps closing",
    "going black",
]
SCREENS = [
    "a pink line down the middle",
    "a black blotch",
    "dead pixels",
    "a cracked display",
    "green tint",
]
ITEMS = ["iPhone X", "Apple Watch Series 3", "AirPods", "MacBook Pro", "iPad Pro", "Apple TV"]
DAMAGES = ["a scratched screen", "dents on the corner", "the seal already broken", "a rattling noise"]

BRAND_REPLIES: dict[str, list[str]] = {
    "ios_update_bug": [
        "@USER We'd like to help. Which iOS version is currently installed? You can find this in Settings > General > About. Also tell us the last step you tried.",
        "@USER Let's check this. After the update, try a force restart (volume up, volume down, then hold the side button). Reply with what happens.",
        "@USER Thanks for writing in. If {app} is crashing on iOS {ios}, please update that app from the App Store and test again. If it continues, we can look at a backup + restore path.",
        "@USER Sorry this update has been rough. Go to Settings > General > iPhone Storage and check whether the update fully installed. Screenshot of Settings > General > About is useful — send via DM if you prefer.",
        "@USER If the device is looping on the Apple logo, connect it to a computer and check for recovery mode. We can walk you through that. Which computer OS do you have?",
    ],
    "hardware_device": [
        "@USER Sorry your {device} is acting up. Try an untested cable and wall adapter that meets Apple spec. If it still will not charge, this likely needs a hardware check — we recommend an Apple Store / authorized service appointment.",
        "@USER Battery drain like that is worth checking. Settings > Battery shows usage. If health is unexpectedly low, a Genius Bar battery diagnostic is the next step. We cannot book that here.",
        "@USER A line or blotch on the display is usually hardware. Please back up the device and book a service appointment. We will not guess whether it is covered until they inspect it.",
        "@USER If the device will not power on, try a force restart. If you only get the Apple logo then black, it needs in-person service. Backup first if iTunes/Finder still sees it.",
        "@USER Overheating while idle is not expected. Remove any case, check for background apps, and if it continues, stop using it and get it inspected — especially if the battery looks swollen.",
    ],
    "apple_id_icloud": [
        "@USER We want to help with your Apple ID. Please DM us the Apple ID email address (not the password) so we can point you to the right iforgot.apple.com steps. We cannot unlock accounts on Twitter.",
        "@USER For a locked Apple ID, start at iforgot.apple.com. If two-factor codes go to a phone you no longer have, you will need account recovery — that can take time and is not something we can bypass here.",
        "@USER Please do not share passwords or security questions on Twitter. Reach us via DM with the Apple ID email only, and we will share the official recovery link.",
        "@USER Sign-in loops after a restore often mean the Apple ID is still in a security lock. Try another network, then iforgot.apple.com. If it still fails, this has to go to account recovery.",
        "@USER If you suspect someone else is accessing the Apple ID, change the password from a trusted device if you still can, and review appleid.apple.com > Devices. DM us if you are fully locked out.",
    ],
    "billing_subscription": [
        "@USER We can help look at a charge, but we cannot issue refunds on Twitter. Please DM the Apple ID email on the receipt (not a screenshot of your full card) and the date of the charge.",
        "@USER Unexpected App Store charges should be reviewed from reportaproblem.apple.com. If a purchase was accidental, start there. We cannot confirm a refund until that form is processed.",
        "@USER For Apple Music / iCloud subscriptions, check Settings > [your name] > Subscriptions. If you already cancelled and were billed again, DM us the Apple ID email so a specialist can review.",
        "@USER Duplicate charges: please keep the receipt emails and file via reportaproblem.apple.com. Do not post full card numbers here. We cannot reverse a bank charge from this channel.",
        "@USER Kid / Family Sharing purchases: the organizer can turn on Ask to Buy. For the charge that already posted, use reportaproblem.apple.com — we cannot refund it in this thread.",
    ],
    "connectivity": [
        "@USER Let's isolate Wi-Fi vs device. Forget the network (Settings > Wi-Fi > i), reboot the router, then rejoin. Does another device stay connected to the same network?",
        "@USER For Bluetooth drops, forget the accessory, toggle Bluetooth, and pair again while the case is open. Which accessory and iOS version are we looking at?",
        "@USER CarPlay disconnects: test with a different cable if wired, or forget the car in Settings > General > CarPlay. Tell us whether it is wired or wireless and the car year.",
        "@USER If Wi-Fi is greyed out, try a force restart. If the toggle stays grey, that can be hardware or a deeper software fault — we may need a support ticket.",
        "@USER AirDrop needs Wi-Fi + Bluetooth on and the receiver set to Contacts Only or Everyone for 10 minutes. Are both devices signed into iCloud?",
    ],
    "backup_storage": [
        "@USER iCloud storage numbers include Mail, backups, and iCloud Drive, not just photo count. Check Settings > [your name] > iCloud > Manage Storage and tell us what the biggest items are.",
        "@USER Photos missing after a restore may still be in iCloud.com > Photos. Please check there before we assume data loss. If they are not, this needs a specialist — do not keep restoring.",
        "@USER Backup failed: Settings > [your name] > iCloud > iCloud Backup. If it says not enough storage, that is the blocker. We cannot expand storage from Twitter; you would change the plan in Settings.",
        "@USER Restore stuck: stay on Wi-Fi and power. If it does not move for several hours, we should look at a computer-based restore. Do you have a Mac or PC available?",
        "@USER We cannot roll iCloud backups backward to last week from this channel. If a newer backup overwrote the old one, a specialist has to review what (if anything) can be recovered.",
    ],
    "order_purchase": [
        "@USER We cannot see your order from this public thread. Please DM the order number and the email used at checkout (not your password) so a specialist can check shipping status.",
        "@USER Damaged delivery: please photograph the box and serial if visible, and DM the order number. Do not continue setup if you want a replacement review. We cannot approve a replacement here.",
        "@USER Pickup emails sometimes lag. Check the Apple Store app under orders, or the Apple ID email. If the store cannot find it, DM us the reservation details.",
        "@USER Address changes are only possible before the order ships, and only through the order specialist. DM the order number — we will not edit addresses on a public tweet.",
        "@USER Missing accessories or a cancelled-but-charged order needs the sales record. Please DM order number + last four of the payment method, never the full card.",
    ],
    "how_to_feature": [
        "@USER You can turn off Wi-Fi Assist in Settings > Cellular > Wi-Fi Assist (near the bottom). That setting is on by default on iOS {ios}.",
        "@USER Do Not Disturb is in Settings > Do Not Disturb, and also in Control Center (moon icon). On iOS {ios} you can schedule it there.",
        "@USER Screenshot: on {device} with Face ID, press side + volume up together. On devices with a Home button, side/top + Home.",
        "@USER Battery usage by app: Settings > Battery. Let it collect for a few hours after a reboot for a fair picture.",
        "@USER Screen recording is in Control Center on iOS {ios}: Settings > Control Center > Customize, add Screen Recording.",
        "@USER Wireless charging depends on the model. iPhone 8 and iPhone X support Qi. Older iPhones do not. We would not assume a third-party charger is certified.",
    ],
    "other": [
        "@USER Thanks for writing in. If you have a specific product issue, send a few details (device + what you see on screen) and we will try to help.",
        "@USER We are here for product support rather than store crowds, carriers, or rumors. If something on your device is broken, tell us the model and iOS version.",
        "@USER We cannot comment on other companies' plans or school filters. For an Apple product question, we are listening.",
        "@USER Thanks. If you need help with an Apple device, reply with the issue and the model.",
    ],
}

PRIOR_CUSTOMER = [
    "@AppleSupport still happening after I restarted",
    "@AppleSupport that did not work",
    "@AppleSupport I already tried that yesterday",
    "@AppleSupport following up — no change",
]
PRIOR_BRAND = [
    "@USER Thanks for the extra detail. We are still looking at this with you.",
    "@USER Sorry it is not resolved yet. Let's try the next step.",
]


def _fill(rng: random.Random, template: str, intent: str) -> str:
    device = rng.choice(DEVICES)
    ios = rng.choice(IOS)
    app = rng.choice(APPS)
    return template.format(
        device=device,
        ios=ios,
        ios_old=rng.choice(["10.3.3", "10.3.2"]),
        app=app,
        extra=rng.choice(EXTRAS),
        symptom=rng.choice(SYMPTOMS),
        screen=rng.choice(SCREENS),
        hours=rng.choice([2, 3, 4, 5]),
        pct=rng.choice([12, 18, 23, 40, 67, 81]),
        months=rng.choice([3, 5, 8, 11]),
        item=rng.choice(ITEMS),
        days=rng.choice([5, 7, 10, 14]),
        damage=rng.choice(DAMAGES),
    )


def _maybe_noisy(rng: random.Random, text: str) -> str:
    if rng.random() < 0.12:
        text = text.replace("Apple", "apple").replace("I ", "i ")
    if rng.random() < 0.08:
        text = text.replace("please", "pls").replace("Please", "Pls")
    if rng.random() < 0.1:
        text = text.replace("n't", "nt").replace("not ", "not ")
    return text


def generate_raw_tweets(
    *,
    per_intent: int = 240,
    seed: int = 42,
    extra_ambiguous: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (tweets_df, gold_map_df). gold_map is conversation_id → gold_intent."""

    rng = random.Random(seed)
    names = [n for n in intent_names() if n in CUSTOMERS]
    tweet_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    tweet_id = 10_000_000
    conv_index = 0

    def add_conversation(intent: str, customer_text: str, brand_text: str, with_prior: bool) -> None:
        nonlocal tweet_id, conv_index
        conv_index += 1
        user = _user(rng)
        created = datetime(2017, 10, 1, tzinfo=timezone.utc) + timedelta(minutes=conv_index * 7)
        wday = calendar.day_abbr[created.weekday()]
        mon = calendar.month_abbr[created.month]

        def fmt(ts: datetime) -> str:
            return f"{calendar.day_abbr[ts.weekday()]} {calendar.month_abbr[ts.month]} {ts.day:02d} {ts:%H:%M:%S} +0000 {ts.year}"

        root_id = tweet_id
        conversation_id = str(root_id)
        prior_brand_id = None
        if with_prior:
            prior_c = tweet_id
            prior_b = tweet_id + 1
            current_c = tweet_id + 2
            current_b = tweet_id + 3
            tweet_id += 4
            t0 = created
            tweet_rows.append(
                {
                    "tweet_id": prior_c,
                    "author_id": user,
                    "inbound": True,
                    "created_at": fmt(t0),
                    "text": rng.choice(PRIOR_CUSTOMER),
                    "response_tweet_id": str(prior_b),
                    "in_response_to_tweet_id": "",
                }
            )
            tweet_rows.append(
                {
                    "tweet_id": prior_b,
                    "author_id": BRAND,
                    "inbound": False,
                    "created_at": fmt(t0 + timedelta(minutes=12)),
                    "text": rng.choice(PRIOR_BRAND),
                    "response_tweet_id": "",
                    "in_response_to_tweet_id": str(prior_c),
                }
            )
            tweet_rows.append(
                {
                    "tweet_id": current_c,
                    "author_id": user,
                    "inbound": True,
                    "created_at": fmt(t0 + timedelta(minutes=40)),
                    "text": customer_text,
                    "response_tweet_id": str(current_b),
                    "in_response_to_tweet_id": str(prior_b),
                }
            )
            tweet_rows.append(
                {
                    "tweet_id": current_b,
                    "author_id": BRAND,
                    "inbound": False,
                    "created_at": fmt(t0 + timedelta(minutes=55)),
                    "text": brand_text,
                    "response_tweet_id": "",
                    "in_response_to_tweet_id": str(current_c),
                }
            )
            gold_rows.append(
                {
                    "conversation_id": str(prior_c),
                    "inbound_tweet_id": str(current_c),
                    "gold_intent": intent,
                }
            )
        else:
            current_c = tweet_id
            current_b = tweet_id + 1
            tweet_id += 2
            tweet_rows.append(
                {
                    "tweet_id": current_c,
                    "author_id": user,
                    "inbound": True,
                    "created_at": fmt(created),
                    "text": customer_text,
                    "response_tweet_id": str(current_b),
                    "in_response_to_tweet_id": "",
                }
            )
            tweet_rows.append(
                {
                    "tweet_id": current_b,
                    "author_id": BRAND,
                    "inbound": False,
                    "created_at": fmt(created + timedelta(minutes=9)),
                    "text": brand_text,
                    "response_tweet_id": "",
                    "in_response_to_tweet_id": str(current_c),
                }
            )
            gold_rows.append(
                {
                    "conversation_id": str(current_c),
                    "inbound_tweet_id": str(current_c),
                    "gold_intent": intent,
                }
            )
        _ = root_id, prior_brand_id, wday, mon

    for intent in names:
        templates = CUSTOMERS[intent]
        replies = BRAND_REPLIES[intent]
        for i in range(per_intent):
            tmpl = templates[i % len(templates)]
            customer = _maybe_noisy(rng, _fill(rng, tmpl, intent))
            reply_tmpl = replies[i % len(replies)]
            brand = _fill(rng, reply_tmpl, intent)
            add_conversation(intent, customer, brand, with_prior=(i % 4 == 0))

    # Ambiguous extras: battery drain AFTER an update (true = ios_update_bug).
    amb_templates = [
        (
            "ios_update_bug",
            "@AppleSupport since the iOS {ios} update my {device} battery is dying by noon. never had this before",
        ),
        (
            "ios_update_bug",
            "@AppleSupport {device} Wi-Fi drops constantly after I installed iOS {ios}. was fine on the old version",
        ),
        (
            "billing_subscription",
            "@AppleSupport I cannot download apps because of a billing problem on my Apple ID. charged me mystery items",
        ),
        (
            "hardware_device",
            "@AppleSupport phone is so slow and hot I thought it was iOS but I have not updated in months. {device}",
        ),
        (
            "backup_storage",
            "@AppleSupport I bought more iCloud storage and still cannot backup. billed me though",
        ),
        (
            "other",
            "@AppleSupport my Uber app crashes. fix your phone. {device}",
        ),
    ]
    for j in range(extra_ambiguous):
        intent, tmpl = amb_templates[j % len(amb_templates)]
        customer = _fill(rng, tmpl, intent)
        brand = _fill(rng, rng.choice(BRAND_REPLIES[intent]), intent)
        add_conversation(intent, customer, brand, with_prior=False)

    tweets = pd.DataFrame(tweet_rows)
    gold = pd.DataFrame(gold_rows)
    return tweets, gold


def write_sample_corpus(per_intent: int = 240, seed: int = 42) -> tuple[str, str]:
    ensure_dirs()
    tweets, gold = generate_raw_tweets(per_intent=per_intent, seed=seed)
    tweets_path = SAMPLE / "tweets.csv"
    gold_path = SAMPLE / "gold_intents.csv"
    tweets.to_csv(tweets_path, index=False)
    gold.to_csv(gold_path, index=False)
    return str(tweets_path), str(gold_path)

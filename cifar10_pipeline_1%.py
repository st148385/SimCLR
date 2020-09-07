import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import time

"""
Erklärung:
Wir können hier mit 'images_per_class = X' angeben, wie viele images es pro Klasse in einem class-balanced way 
geben soll.
Dann zählt eine Schleife die Bilder pro Klasse des ungeshufflten cifar10 training-sets, bis 'X' für jede Klasse erreicht
ist. Alle dementsprechend 'gewollten' Images werden in die Split.txt geschrieben und alle überflüssigen (Images die die 
class-balance verhindern würden) werden vom Split ausgenommen.
Das meiste dieser .py Datei sind die nochmalige Überprüfung, ob die Splits tatsächlich korrekt sind.
"""

data, info = tfds.load(name='cifar10',
                       data_dir='~\\tensorflow_datasets',
                       split=tfds.Split.TRAIN,      #First 10 labels: 7 8 4 4 6 5 2 9 6 6
                       shuffle_files=False,
                       with_info=True)


def _map_data(*args):
    image = args[0]['image']
    label = args[0]['label']
    # label = tf.one_hot(label, info.features['label'].num_classes)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


dataset = data.map(map_func=_map_data)

class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0
class5 = 0
class6 = 0
class7 = 0
class8 = 0
class9 = 0

list_of_unwanted = []

i = 0

images_per_class = 50       # N% Subset: images_per_class = 50000 * N/100 * 1/10

for image, label in dataset.take(50000):

    if class0 >= images_per_class and class1 >= images_per_class and class2 >= images_per_class and class3 >= images_per_class and class4 >= images_per_class and class5 >= images_per_class and class6 >= images_per_class and class7 >= images_per_class and class8 >= images_per_class and class9 >= images_per_class:
        print(f"breaking the for-loop now! (at i={i})")
        break
    else:
        if label == 0:
            class0 += 1
            if class0 > images_per_class:
                list_of_unwanted.append(i)
        elif label == 1:
            class1 += 1
            if class1 > images_per_class:
                list_of_unwanted.append(i)
        elif label == 2:
            class2 += 1
            if class2 > images_per_class:
                list_of_unwanted.append(i)
        elif label == 3:
            class3 += 1
            if class3 > images_per_class:
                list_of_unwanted.append(i)
        elif label == 4:
            class4 += 1
            if class4 > images_per_class:
                list_of_unwanted.append(i)
        elif label == 5:
            class5 += 1
            if class5 > images_per_class:
                list_of_unwanted.append(i)
        elif label == 6:
            class6 += 1
            if class6 > images_per_class:
                list_of_unwanted.append(i)
        elif label == 7:
            class7 += 1
            if class7 > images_per_class:
                list_of_unwanted.append(i)
        elif label == 8:
            class8 += 1
            if class8 > images_per_class:
                list_of_unwanted.append(i)
        elif label == 9:
            class9 += 1
            if class9 > images_per_class:
                list_of_unwanted.append(i)
        i += 1
        print("did iteration", i)

print(class0, class1, class2, class3, class4, class5, class6, class7, class8, class9)
sum = class0 + class1 + class2 + class3 + class4 + class5 + class6 + class7 + class8 + class9
print("sum:", sum)

print(list_of_unwanted)
print('Length of this list:', len(list_of_unwanted))

file = open("cifar10_N%_Split.txt", "w+")
file.write("train[:%d]+" % list_of_unwanted[0])
for r in range(len(list_of_unwanted) - 2):
    if list_of_unwanted[r] + 1 != list_of_unwanted[r + 1]:
        file.write("train[%d:%d]+" % (list_of_unwanted[r] + 1, list_of_unwanted[r + 1]))
file.write("train[%d:%d]" % (list_of_unwanted[-1] + 1, sum))
file.close()

"""Ab hier nur noch double-check, ob die Splits genau richtig sind."""

# Check if correct:
print("Checking the 1% split:")
(train_data, val_data), info = tfds.load(name='cifar10',
                             data_dir='~\\tensorflow_datasets',
                             split=['train[:374]+train[375:383]+train[384:388]+train[389:390]+train[391:394]+train['
                                   '396:398]+train[399:400]+train[401:403]+train[404:408]+train[409:410]+train['
                                   '413:419]+train[421:423]+train[424:436]+train[438:439]+train[440:442]+train['
                                   '443:447]+train[448:453]+train[454:459]+train[461:462]+train[463:465]+train['
                                   '466:467]+train[468:469]+train[475:477]+train[478:481]+train[483:484]+train['
                                   '488:490]+train[491:493]+train[494:495]+train[496:498]+train[499:500]+train['
                                   '501:502]+train[503:505]+train[508:509]+train[510:512]+train[514:519]+train['
                                   '524:525]+train[529:531]+train[532:533]+train[535:536]+train[537:539]+train['
                                   '545:547]+train[548:550]+train[553:555]+train[560:563]+train[567:569]+train['
                                   '581:582]+train[584:585]+train[587:588]+train[589:592]+train[595:596]+train['
                                   '597:599]+train[601:604]+train[610:611]+train[624:625]+train[658:659]','test'],
                             shuffle_files=False,
                             with_info=True)


def _map_data(*args):
    image = args[0]['image']
    label = args[0]['label']
    # label = tf.one_hot(label, info.features['label'].num_classes)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


ds_neu = train_data.map(map_func=_map_data)

# Show first 10 labels of ds_neu
# for image, label in ds_neu.take(10):
#     print(label.numpy())

class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0
class5 = 0
class6 = 0
class7 = 0
class8 = 0
class9 = 0

list_of_unwanted = []

i = 0

for image, label in ds_neu.take(659):
    if label == 0:
        class0 += 1
        if class0 > 50:
            list_of_unwanted.append(i)
    elif label == 1:
        class1 += 1
        if class1 > 50:
            list_of_unwanted.append(i)
    elif label == 2:
        class2 += 1
        if class2 > 50:
            list_of_unwanted.append(i)
    elif label == 3:
        class3 += 1
        if class3 > 50:
            list_of_unwanted.append(i)
    elif label == 4:
        class4 += 1
        if class4 > 50:
            list_of_unwanted.append(i)
    elif label == 5:
        class5 += 1
        if class5 > 50:
            list_of_unwanted.append(i)
    elif label == 6:
        class6 += 1
        if class6 > 50:
            list_of_unwanted.append(i)
    elif label == 7:
        class7 += 1
        if class7 > 50:
            list_of_unwanted.append(i)
    elif label == 8:
        class8 += 1
        if class8 > 50:
            list_of_unwanted.append(i)
    elif label == 9:
        class9 += 1
        if class9 > 50:
            list_of_unwanted.append(i)
    i += 1
print(class0, class1, class2, class3, class4, class5, class6, class7, class8, class9)
print("sum:", class0 + class1 + class2 + class3 + class4 + class5 + class6 + class7 + class8 + class9)

print(list_of_unwanted)

# Did the same for 10% split. Checking if its correct now:
print("Checking the 10% split:")
t0 = time.process_time()    #Check duration of tfds.load (print is below tfds.load)
(train_data, test_data), info = tfds.load(name='cifar10',
                                          data_dir='~\\tensorflow_datasets',
                                          split=[
                                              'train[:4723]+train[4725:4728]+train[4729:4730]+train[4731:4735]+train['
                                              '4736:4742]+train[4743:4746]+train[4747:4762]+train[4764:4768]+train['
                                              '4769:4782]+train[4783:4786]+train[4787:4792]+train[4793:4802]+train['
                                              '4803:4810]+train[4811:4814]+train[4815:4819]+train[4820:4821]+train['
                                              '4822:4823]+train[4824:4837]+train[4839:4841]+train[4842:4844]+train['
                                              '4845:4847]+train[4848:4849]+train[4850:4856]+train[4857:4863]+train['
                                              '4864:4867]+train[4868:4876]+train[4878:4885]+train[4886:4889]+train['
                                              '4891:4893]+train[4895:4896]+train[4897:4898]+train[4900:4901]+train['
                                              '4905:4910]+train[4911:4914]+train[4915:4921]+train[4922:4923]+train['
                                              '4925:4926]+train[4928:4930]+train[4931:4933]+train[4938:4939]+train['
                                              '4940:4941]+train[4942:4948]+train[4950:4952]+train[4953:4954]+train['
                                              '4955:4956]+train[4957:4959]+train[4960:4962]+train[4963:4964]+train['
                                              '4965:4970]+train[4971:4972]+train[4975:4978]+train[4980:4984]+train['
                                              '4987:4989]+train[4993:4996]+train[4997:4998]+train[4999:5001]+train['
                                              '5003:5006]+train[5007:5008]+train[5010:5012]+train[5013:5014]+train['
                                              '5016:5018]+train[5019:5021]+train[5023:5024]+train[5025:5026]+train['
                                              '5029:5030]+train[5032:5033]+train[5035:5036]+train[5037:5038]+train['
                                              '5039:5041]+train[5042:5043]+train[5048:5049]+train[5050:5051]+train['
                                              '5052:5053]+train[5054:5055]+train[5059:5061]+train[5065:5066]+train['
                                              '5069:5070]+train[5071:5072]+train[5073:5074]+train[5075:5078]+train['
                                              '5089:5092]+train[5094:5095]+train[5098:5099]+train[5102:5103]+train['
                                              '5104:5106]+train[5107:5109]+train[5110:5111]+train[5113:5114]+train['
                                              '5115:5116]+train[5119:5120]+train[5129:5132]+train[5136:5137]+train['
                                              '5145:5147]+train[5148:5149]+train[5155:5156]+train[5157:5158]+train['
                                              '5165:5166]+train[5171:5172]+train[5180:5181]+train[5186:5187]+train['
                                              '5197:5198]+train[5202:5203]+train[5204:5207]+train[5214:5216]+train['
                                              '5230:5231]+train[5248:5249]+train[5257:5258]+train[5266:5267]+train['
                                              '5268:5269]+train[5270:5272]+train[5273:5275]+train[5280:5281]+train['
                                              '5283:5284]+train[5288:5289]', 'test'],
                                          shuffle_files=False,
                                          with_info=True)
t1 = time.process_time() - t0
print(f"Time elapsed: {t1}seconds (CPU seconds elapsed (floating point))")


def _map_data(*args):
    image = args[0]['image']
    label = args[0]['label']
    # label = tf.one_hot(label, info.features['label'].num_classes)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


ds_neu = train_data.map(map_func=_map_data)

class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0
class5 = 0
class6 = 0
class7 = 0
class8 = 0
class9 = 0

list_of_unwanted = []

i = 0

for image, label in ds_neu.take(6000):
    if label == 0:
        class0 += 1
        if class0 > 500:
            list_of_unwanted.append(i)
    elif label == 1:
        class1 += 1
        if class1 > 500:
            list_of_unwanted.append(i)
    elif label == 2:
        class2 += 1
        if class2 > 500:
            list_of_unwanted.append(i)
    elif label == 3:
        class3 += 1
        if class3 > 500:
            list_of_unwanted.append(i)
    elif label == 4:
        class4 += 1
        if class4 > 500:
            list_of_unwanted.append(i)
    elif label == 5:
        class5 += 1
        if class5 > 500:
            list_of_unwanted.append(i)
    elif label == 6:
        class6 += 1
        if class6 > 500:
            list_of_unwanted.append(i)
    elif label == 7:
        class7 += 1
        if class7 > 500:
            list_of_unwanted.append(i)
    elif label == 8:
        class8 += 1
        if class8 > 500:
            list_of_unwanted.append(i)
    elif label == 9:
        class9 += 1
        if class9 > 500:
            list_of_unwanted.append(i)
    i += 1
print(class0, class1, class2, class3, class4, class5, class6, class7, class8, class9)
print("sum:", class0 + class1 + class2 + class3 + class4 + class5 + class6 + class7 + class8 + class9)

print(list_of_unwanted)

# Did the same for 20% split. Checking if its correct now:
print("Checking the 20% split:")
t0 = time.process_time()    #Check duration of tfds.load (print is below tfds.load)
(train_data, test_data), info = tfds.load(name='cifar10',
                                          data_dir='~\\tensorflow_datasets',
                                          split=[
                                              'train[:9539]+train[9540:9553]+train[9554:9560]+train[9561:9577]+train['
                                              '9578:9580]+train[9581:9594]+train[9595:9612]+train[9613:9623]+train['
                                              '9624:9628]+train[9629:9633]+train[9635:9639]+train[9640:9656]+train['
                                              '9657:9663]+train[9664:9667]+train[9669:9678]+train[9679:9716]+train['
                                              '9717:9728]+train[9729:9733]+train[9734:9735]+train[9736:9744]+train['
                                              '9745:9755]+train[9756:9759]+train[9760:9766]+train[9767:9770]+train['
                                              '9771:9773]+train[9774:9778]+train[9782:9785]+train[9786:9787]+train['
                                              '9789:9794]+train[9795:9799]+train[9800:9803]+train[9805:9815]+train['
                                              '9816:9818]+train[9820:9824]+train[9825:9830]+train[9831:9836]+train['
                                              '9839:9841]+train[9844:9849]+train[9850:9851]+train[9852:9857]+train['
                                              '9858:9863]+train[9864:9865]+train[9866:9867]+train[9868:9871]+train['
                                              '9872:9873]+train[9874:9875]+train[9876:9879]+train[9881:9886]+train['
                                              '9887:9889]+train[9890:9899]+train[9900:9903]+train[9906:9907]+train['
                                              '9908:9913]+train[9915:9916]+train[9917:9924]+train[9925:9928]+train['
                                              '9929:9930]+train[9931:9933]+train[9935:9937]+train[9938:9940]+train['
                                              '9941:9942]+train[9943:9945]+train[9946:9948]+train[9949:9950]+train['
                                              '9951:9953]+train[9954:9956]+train[9958:9962]+train[9963:9967]+train['
                                              '9968:9970]+train[9971:9972]+train[9974:9976]+train[9977:9980]+train['
                                              '9981:9982]+train[9983:9985]+train[9986:9987]+train[9988:9991]+train['
                                              '9992:9994]+train[9995:9996]+train[9997:9998]+train[10003:10005]+train['
                                              '10006:10007]+train[10008:10009]+train[10010:10012]+train['
                                              '10013:10015]+train[10019:10020]+train[10022:10024]+train['
                                              '10025:10027]+train[10029:10032]+train[10033:10034]+train['
                                              '10035:10037]+train[10039:10043]+train[10044:10048]+train['
                                              '10050:10051]+train[10052:10058]+train[10060:10062]+train['
                                              '10063:10073]+train[10074:10076]+train[10078:10079]+train['
                                              '10080:10081]+train[10082:10086]+train[10093:10094]+train['
                                              '10097:10098]+train[10103:10104]+train[10106:10107]+train['
                                              '10109:10111]+train[10114:10115]+train[10116:10118]+train['
                                              '10120:10121]+train[10123:10125]+train[10128:10129]+train['
                                              '10137:10138]+train[10147:10150]+train[10151:10155]+train['
                                              '10160:10161]+train[10162:10163]+train[10164:10165]+train['
                                              '10166:10167]+train[10168:10169]+train[10170:10171]+train['
                                              '10174:10175]+train[10176:10177]+train[10182:10183]+train['
                                              '10186:10187]+train[10194:10196]+train[10197:10198]+train['
                                              '10211:10212]+train[10220:10222]+train[10223:10227]+train['
                                              '10230:10231]+train[10239:10240]+train[10243:10244]+train['
                                              '10249:10250]', 'test'],
                                          shuffle_files=False,
                                          with_info=True)
t1 = time.process_time() - t0
print(f"Time elapsed: {t1}seconds (CPU seconds elapsed (floating point))")


def _map_data(*args):
    image = args[0]['image']
    label = args[0]['label']
    # label = tf.one_hot(label, info.features['label'].num_classes)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


ds_neu = train_data.map(map_func=_map_data)

class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0
class5 = 0
class6 = 0
class7 = 0
class8 = 0
class9 = 0

list_of_unwanted = []

i = 0

for image, label in ds_neu.take(12000):
    if label == 0:
        class0 += 1
        if class0 > 1000:
            list_of_unwanted.append(i)
    elif label == 1:
        class1 += 1
        if class1 > 1000:
            list_of_unwanted.append(i)
    elif label == 2:
        class2 += 1
        if class2 > 1000:
            list_of_unwanted.append(i)
    elif label == 3:
        class3 += 1
        if class3 > 1000:
            list_of_unwanted.append(i)
    elif label == 4:
        class4 += 1
        if class4 > 1000:
            list_of_unwanted.append(i)
    elif label == 5:
        class5 += 1
        if class5 > 1000:
            list_of_unwanted.append(i)
    elif label == 6:
        class6 += 1
        if class6 > 1000:
            list_of_unwanted.append(i)
    elif label == 7:
        class7 += 1
        if class7 > 1000:
            list_of_unwanted.append(i)
    elif label == 8:
        class8 += 1
        if class8 > 1000:
            list_of_unwanted.append(i)
    elif label == 9:
        class9 += 1
        if class9 > 1000:
            list_of_unwanted.append(i)
    i += 1
print(class0, class1, class2, class3, class4, class5, class6, class7, class8, class9)
print("sum:", class0 + class1 + class2 + class3 + class4 + class5 + class6 + class7 + class8 + class9)

print(list_of_unwanted)


# Did the same for 5% split. Checking if its correct now:
print("Checking the 5% split:")
t0 = time.process_time()    #Check duration of tfds.load (print is below tfds.load)
(train_data, test_data), info = tfds.load(name='cifar10',
                                          data_dir='~\\tensorflow_datasets',
                                          split=[
                                              'train[:2333]+train[2334:2346]+train[2347:2356]+train[2357:2366]+train['
                                              '2370:2376]+train[2377:2379]+train[2380:2383]+train[2385:2390]+train['
                                              '2391:2392]+train[2393:2396]+train[2397:2401]+train[2402:2403]+train['
                                              '2404:2408]+train[2409:2410]+train[2411:2413]+train[2414:2418]+train['
                                              '2419:2422]+train[2424:2425]+train[2427:2430]+train[2432:2435]+train['
                                              '2436:2442]+train[2445:2449]+train[2450:2454]+train[2455:2457]+train['
                                              '2458:2459]+train[2460:2468]+train[2471:2476]+train[2477:2478]+train['
                                              '2480:2483]+train[2484:2486]+train[2488:2492]+train[2495:2497]+train['
                                              '2498:2501]+train[2506:2511]+train[2516:2517]+train[2522:2523]+train['
                                              '2526:2527]+train[2532:2533]+train[2539:2543]+train[2544:2546]+train['
                                              '2549:2550]+train[2552:2553]+train[2554:2555]+train[2558:2559]+train['
                                              '2560:2561]+train[2563:2566]+train[2567:2569]+train[2576:2577]+train['
                                              '2578:2582]+train[2588:2589]+train[2592:2593]+train[2613:2615]+train['
                                              '2618:2619]+train[2620:2621]+train[2622:2623]+train[2629:2630]+train['
                                              '2634:2635]+train[2637:2638]+train[2639:2640]+train[2641:2642]+train['
                                              '2645:2646]+train[2658:2659]+train[2690:2691]+train[2702:2703]', 'test'],
                                          shuffle_files=False,
                                          with_info=True)
t1 = time.process_time() - t0
print(f"Time elapsed: {t1}seconds (CPU seconds elapsed (floating point))")


def _map_data(*args):
    image = args[0]['image']
    label = args[0]['label']
    # label = tf.one_hot(label, info.features['label'].num_classes)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


ds_neu = train_data.map(map_func=_map_data)

class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0
class5 = 0
class6 = 0
class7 = 0
class8 = 0
class9 = 0

list_of_unwanted = []

i = 0

for image, label in ds_neu.take(6000):
    if label == 0:
        class0 += 1
        if class0 > 250:
            list_of_unwanted.append(i)
    elif label == 1:
        class1 += 1
        if class1 > 250:
            list_of_unwanted.append(i)
    elif label == 2:
        class2 += 1
        if class2 > 250:
            list_of_unwanted.append(i)
    elif label == 3:
        class3 += 1
        if class3 > 250:
            list_of_unwanted.append(i)
    elif label == 4:
        class4 += 1
        if class4 > 250:
            list_of_unwanted.append(i)
    elif label == 5:
        class5 += 1
        if class5 > 250:
            list_of_unwanted.append(i)
    elif label == 6:
        class6 += 1
        if class6 > 250:
            list_of_unwanted.append(i)
    elif label == 7:
        class7 += 1
        if class7 > 250:
            list_of_unwanted.append(i)
    elif label == 8:
        class8 += 1
        if class8 > 250:
            list_of_unwanted.append(i)
    elif label == 9:
        class9 += 1
        if class9 > 250:
            list_of_unwanted.append(i)
    i += 1
print(class0, class1, class2, class3, class4, class5, class6, class7, class8, class9)
print("sum:", class0 + class1 + class2 + class3 + class4 + class5 + class6 + class7 + class8 + class9)

print(list_of_unwanted)


# Did the same for 30% split. Checking if its correct now:
print("Checking the 30% split:")
t0 = time.process_time()    #Check duration of tfds.load (print is below tfds.load)
(train_data, test_data), info = tfds.load(name='cifar10',
                  data_dir='~\\tensorflow_datasets',
                  split=['train[:14543]+train[14544:14565]+train[14566:14576]+train[14577:14616]+train['
                         '14617:14627]+train[14628:14629]+train[14630:14664]+train[14665:14666]+train['
                         '14668:14671]+train[14673:14680]+train[14681:14689]+train[14690:14691]+train['
                         '14692:14693]+train[14694:14707]+train[14708:14713]+train[14714:14723]+train['
                         '14724:14731]+train[14732:14736]+train[14737:14747]+train[14748:14761]+train['
                         '14762:14771]+train[14773:14783]+train[14784:14785]+train[14787:14788]+train['
                         '14789:14795]+train[14796:14805]+train[14806:14808]+train[14810:14811]+train['
                         '14812:14821]+train[14822:14829]+train[14830:14832]+train[14834:14846]+train['
                         '14847:14850]+train[14852:14854]+train[14855:14860]+train[14861:14862]+train['
                         '14866:14869]+train[14870:14873]+train[14874:14875]+train[14876:14877]+train['
                         '14879:14880]+train[14887:14890]+train[14891:14892]+train[14893:14900]+train['
                         '14901:14914]+train[14917:14922]+train[14923:14925]+train[14926:14927]+train['
                         '14928:14929]+train[14930:14932]+train[14933:14940]+train[14945:14947]+train['
                         '14948:14949]+train[14950:14953]+train[14956:14958]+train[14959:14962]+train['
                         '14964:14968]+train[14969:14971]+train[14972:14973]+train[14974:14977]+train['
                         '14979:14981]+train[14982:14984]+train[14985:14988]+train[14989:14990]+train['
                         '14991:14992]+train[14993:14994]+train[14995:14999]+train[15001:15002]+train['
                         '15004:15006]+train[15011:15016]+train[15019:15021]+train[15024:15025]+train['
                         '15026:15028]+train[15029:15031]+train[15032:15033]+train[15037:15038]+train['
                         '15041:15042]+train[15045:15047]+train[15049:15052]+train[15053:15054]+train['
                         '15055:15056]+train[15057:15060]+train[15061:15062]+train[15064:15068]+train['
                         '15074:15075]+train[15076:15077]+train[15079:15080]+train[15081:15082]+train['
                         '15085:15086]+train[15087:15088]+train[15089:15092]+train[15093:15094]+train['
                         '15095:15100]+train[15101:15102]+train[15107:15109]+train[15110:15111]+train['
                         '15112:15113]+train[15126:15128]+train[15132:15133]+train[15148:15149]+train['
                         '15151:15152]+train[15155:15156]+train[15159:15160]+train[15162:15163]+train['
                         '15174:15175]+train[15179:15180]+train[15197:15198]+train[15241:15242]+train['
                         '15261:15262]+train[15271:15272]+train[15276:15277]+train[15288:15290]+train['
                         '15292:15293]+train[15299:15300]+train[15313:15314]+train[15335:15337]+train['
                         '15339:15341]+train[15357:15358]+train[15368:15370]+train[15374:15375]+train['
                         '15384:15385]+train[15386:15387]+train[15388:15389]+train[15395:15396]+train['
                         '15402:15403]+train[15406:15407]+train[15411:15412]+train[15417:15418]+train['
                         '15445:15446]+train[15452:15453]+train[15469:15470]+train[15471:15472]+train[15477:15478]',
                         'test'],
                  shuffle_files=False,
                  with_info=True)
t1 = time.process_time() - t0
print(f"Time elapsed: {t1}seconds (CPU seconds elapsed (floating point))")


def _map_data(*args):
    image = args[0]['image']
    label = args[0]['label']
    # label = tf.one_hot(label, info.features['label'].num_classes)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


ds_neu = train_data.map(map_func=_map_data)

class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0
class5 = 0
class6 = 0
class7 = 0
class8 = 0
class9 = 0

list_of_unwanted = []

i = 0

for image, label in ds_neu.take(50000):
    if label == 0:
        class0 += 1
        if class0 > 1500:
            list_of_unwanted.append(i)
    elif label == 1:
        class1 += 1
        if class1 > 1500:
            list_of_unwanted.append(i)
    elif label == 2:
        class2 += 1
        if class2 > 1500:
            list_of_unwanted.append(i)
    elif label == 3:
        class3 += 1
        if class3 > 1500:
            list_of_unwanted.append(i)
    elif label == 4:
        class4 += 1
        if class4 > 1500:
            list_of_unwanted.append(i)
    elif label == 5:
        class5 += 1
        if class5 > 1500:
            list_of_unwanted.append(i)
    elif label == 6:
        class6 += 1
        if class6 > 1500:
            list_of_unwanted.append(i)
    elif label == 7:
        class7 += 1
        if class7 > 1500:
            list_of_unwanted.append(i)
    elif label == 8:
        class8 += 1
        if class8 > 1500:
            list_of_unwanted.append(i)
    elif label == 9:
        class9 += 1
        if class9 > 1500:
            list_of_unwanted.append(i)
    i += 1
print(class0, class1, class2, class3, class4, class5, class6, class7, class8, class9)
print("sum:", class0 + class1 + class2 + class3 + class4 + class5 + class6 + class7 + class8 + class9)

print(list_of_unwanted)


# Did the same for 35% split. Checking if its correct now:
print("Checking the 35% split:")
t0 = time.process_time()    #Check duration of tfds.load (print is below tfds.load)
(train_data, test_data), info = tfds.load(name='cifar10',
                  data_dir='~\\tensorflow_datasets',
                  split=['train[:17045]+train[17046:17056]+train[17057:17064]+train[17065:17075]+train['
                         '17076:17077]+train[17078:17095]+train[17096:17097]+train[17098:17106]+train['
                         '17107:17112]+train[17113:17116]+train[17117:17128]+train[17129:17131]+train['
                         '17132:17134]+train[17135:17140]+train[17141:17142]+train[17143:17145]+train['
                         '17146:17170]+train[17171:17172]+train[17174:17175]+train[17176:17181]+train['
                         '17182:17189]+train[17190:17207]+train[17208:17210]+train[17211:17214]+train['
                         '17215:17216]+train[17217:17218]+train[17219:17221]+train[17222:17223]+train['
                         '17224:17227]+train[17228:17229]+train[17230:17242]+train[17245:17250]+train['
                         '17251:17259]+train[17260:17264]+train[17266:17267]+train[17268:17274]+train['
                         '17277:17278]+train[17279:17281]+train[17282:17286]+train[17288:17295]+train['
                         '17296:17298]+train[17299:17302]+train[17303:17318]+train[17319:17321]+train['
                         '17322:17323]+train[17324:17328]+train[17329:17331]+train[17332:17333]+train['
                         '17334:17335]+train[17337:17338]+train[17339:17344]+train[17346:17347]+train['
                         '17350:17362]+train[17363:17364]+train[17365:17366]+train[17370:17372]+train['
                         '17373:17375]+train[17376:17377]+train[17379:17381]+train[17383:17384]+train['
                         '17386:17389]+train[17390:17395]+train[17397:17401]+train[17402:17403]+train['
                         '17404:17406]+train[17408:17411]+train[17413:17415]+train[17416:17420]+train['
                         '17421:17422]+train[17423:17425]+train[17426:17427]+train[17428:17436]+train['
                         '17437:17439]+train[17441:17442]+train[17444:17445]+train[17446:17447]+train['
                         '17448:17452]+train[17453:17454]+train[17455:17456]+train[17459:17461]+train['
                         '17462:17464]+train[17465:17467]+train[17469:17474]+train[17475:17476]+train['
                         '17482:17483]+train[17484:17485]+train[17487:17492]+train[17493:17494]+train['
                         '17495:17498]+train[17499:17501]+train[17502:17503]+train[17504:17506]+train['
                         '17508:17512]+train[17513:17515]+train[17516:17520]+train[17521:17523]+train['
                         '17524:17529]+train[17532:17533]+train[17534:17537]+train[17540:17543]+train['
                         '17546:17549]+train[17559:17561]+train[17562:17563]+train[17568:17569]+train['
                         '17570:17571]+train[17573:17574]+train[17578:17579]+train[17581:17583]+train['
                         '17584:17585]+train[17588:17589]+train[17593:17594]+train[17595:17596]+train['
                         '17602:17606]+train[17607:17610]+train[17611:17613]+train[17614:17615]+train['
                         '17616:17618]+train[17619:17620]+train[17621:17623]+train[17628:17630]+train['
                         '17631:17633]+train[17644:17645]+train[17652:17653]+train[17654:17655]+train['
                         '17662:17663]+train[17665:17666]+train[17667:17669]+train[17670:17671]+train['
                         '17685:17686]+train[17689:17690]+train[17692:17693]+train[17698:17700]+train['
                         '17703:17704]+train[17706:17707]+train[17710:17711]+train[17727:17728]+train['
                         '17733:17734]+train[17737:17738]+train[17741:17743]+train[17747:17749]+train['
                         '17753:17755]+train[17761:17763]+train[17768:17770]+train[17773:17774]+train['
                         '17832:17833]+train[17843:17844]+train[17855:17856]+train[17860:17861]+train['
                         '17886:17887]+train[17898:17899]+train[17921:17922]+train[17941:17943]+train['
                         '17949:17950]+train[17954:17955]+train[17970:17971]+train[17979:17980]+train['
                         '18001:18002]+train[18008:18009]+train[18014:18015]+train[18027:18028]+train['
                         '18036:18038]+train[18044:18045]+train[18059:18060]+train[18091:18092]+train['
                         '18100:18102]+train[18134:18135]+train[18140:18141]+train[18155:18156]',
                         'test'],
                  shuffle_files=False,
                  with_info=True)
t1 = time.process_time() - t0
print(f"Time elapsed: {t1}seconds (CPU seconds elapsed (floating point))")


def _map_data(*args):
    image = args[0]['image']
    label = args[0]['label']
    # label = tf.one_hot(label, info.features['label'].num_classes)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


ds_neu = train_data.map(map_func=_map_data)

class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0
class5 = 0
class6 = 0
class7 = 0
class8 = 0
class9 = 0

list_of_unwanted = []

i = 0

for image, label in ds_neu.take(50000):
    if label == 0:
        class0 += 1
        if class0 > 1750:
            list_of_unwanted.append(i)
    elif label == 1:
        class1 += 1
        if class1 > 1750:
            list_of_unwanted.append(i)
    elif label == 2:
        class2 += 1
        if class2 > 1750:
            list_of_unwanted.append(i)
    elif label == 3:
        class3 += 1
        if class3 > 1750:
            list_of_unwanted.append(i)
    elif label == 4:
        class4 += 1
        if class4 > 1750:
            list_of_unwanted.append(i)
    elif label == 5:
        class5 += 1
        if class5 > 1750:
            list_of_unwanted.append(i)
    elif label == 6:
        class6 += 1
        if class6 > 1750:
            list_of_unwanted.append(i)
    elif label == 7:
        class7 += 1
        if class7 > 1750:
            list_of_unwanted.append(i)
    elif label == 8:
        class8 += 1
        if class8 > 1750:
            list_of_unwanted.append(i)
    elif label == 9:
        class9 += 1
        if class9 > 1750:
            list_of_unwanted.append(i)
    i += 1
print(class0, class1, class2, class3, class4, class5, class6, class7, class8, class9)
print("sum:", class0 + class1 + class2 + class3 + class4 + class5 + class6 + class7 + class8 + class9)

print(list_of_unwanted)

# Did the same for 50% split. Checking if its correct now:
print("Checking the 50% split:")
t0 = time.process_time()    #Check duration of tfds.load (print is below tfds.load)
(train_data, test_data), info = tfds.load(name='cifar10',
                  data_dir='~\\tensorflow_datasets',
                  split=['train[:24408]+train[24409:24410]+train[24411:24420]+train[24421:24423]+train['
                         '24424:24429]+train[24430:24435]+train[24437:24439]+train[24440:24449]+train['
                         '24450:24451]+train[24452:24457]+train[24458:24460]+train[24461:24466]+train['
                         '24467:24476]+train[24477:24479]+train[24480:24488]+train[24490:24517]+train['
                         '24518:24529]+train[24530:24538]+train[24539:24549]+train[24552:24553]+train['
                         '24555:24561]+train[24562:24577]+train[24578:24579]+train[24580:24587]+train['
                         '24588:24595]+train[24596:24615]+train[24616:24619]+train[24620:24636]+train['
                         '24637:24646]+train[24647:24689]+train[24690:24703]+train[24704:24731]+train['
                         '24732:24749]+train[24750:24763]+train[24765:24766]+train[24767:24771]+train['
                         '24773:24774]+train[24775:24779]+train[24780:24787]+train[24788:24790]+train['
                         '24791:24794]+train[24795:24802]+train[24804:24816]+train[24817:24823]+train['
                         '24824:24831]+train[24832:24840]+train[24841:24844]+train[24846:24847]+train['
                         '24848:24849]+train[24850:24853]+train[24854:24859]+train[24860:24871]+train['
                         '24873:24874]+train[24877:24887]+train[24888:24892]+train[24894:24895]+train['
                         '24896:24900]+train[24903:24904]+train[24905:24907]+train[24908:24918]+train['
                         '24919:24924]+train[24925:24926]+train[24928:24931]+train[24932:24933]+train['
                         '24934:24935]+train[24938:24941]+train[24942:24946]+train[24948:24950]+train['
                         '24952:24953]+train[24955:24956]+train[24959:24960]+train[24961:24964]+train['
                         '24967:24971]+train[24972:24975]+train[24977:24979]+train[24980:24987]+train['
                         '24988:24989]+train[24990:24993]+train[24994:24995]+train[24997:24999]+train['
                         '25003:25004]+train[25005:25008]+train[25009:25014]+train[25017:25023]+train['
                         '25025:25026]+train[25029:25030]+train[25033:25034]+train[25035:25036]+train['
                         '25037:25041]+train[25042:25044]+train[25045:25046]+train[25047:25050]+train['
                         '25051:25053]+train[25059:25060]+train[25062:25065]+train[25068:25069]+train['
                         '25070:25073]+train[25077:25078]+train[25080:25082]+train[25084:25086]+train['
                         '25089:25090]+train[25093:25094]+train[25096:25097]+train[25102:25104]+train['
                         '25105:25107]+train[25114:25115]+train[25117:25118]+train[25119:25120]+train['
                         '25121:25122]+train[25126:25130]+train[25132:25133]+train[25134:25135]+train['
                         '25139:25141]+train[25142:25144]+train[25145:25147]+train[25148:25149]+train['
                         '25152:25154]+train[25158:25160]+train[25161:25162]+train[25164:25165]+train['
                         '25168:25170]+train[25172:25174]+train[25176:25178]+train[25182:25184]+train['
                         '25188:25189]+train[25190:25192]+train[25195:25197]+train[25199:25200]+train['
                         '25206:25207]+train[25210:25213]+train[25217:25218]+train[25219:25220]+train['
                         '25221:25223]+train[25226:25227]+train[25230:25232]+train[25233:25234]+train['
                         '25236:25237]+train[25238:25239]+train[25245:25246]+train[25253:25254]+train['
                         '25258:25259]+train[25261:25262]+train[25265:25266]',
                         'test'],
                  shuffle_files=False,
                  with_info=True)
t1 = time.process_time() - t0
print(f"Time elapsed: {t1}seconds (CPU seconds elapsed (floating point))")


def _map_data(*args):
    image = args[0]['image']
    label = args[0]['label']
    # label = tf.one_hot(label, info.features['label'].num_classes)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


ds_neu = train_data.map(map_func=_map_data)

class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0
class5 = 0
class6 = 0
class7 = 0
class8 = 0
class9 = 0

list_of_unwanted = []

i = 0

for image, label in ds_neu.take(50000):
    if label == 0:
        class0 += 1
        if class0 > 2500:
            list_of_unwanted.append(i)
    elif label == 1:
        class1 += 1
        if class1 > 2500:
            list_of_unwanted.append(i)
    elif label == 2:
        class2 += 1
        if class2 > 2500:
            list_of_unwanted.append(i)
    elif label == 3:
        class3 += 1
        if class3 > 2500:
            list_of_unwanted.append(i)
    elif label == 4:
        class4 += 1
        if class4 > 2500:
            list_of_unwanted.append(i)
    elif label == 5:
        class5 += 1
        if class5 > 2500:
            list_of_unwanted.append(i)
    elif label == 6:
        class6 += 1
        if class6 > 2500:
            list_of_unwanted.append(i)
    elif label == 7:
        class7 += 1
        if class7 > 2500:
            list_of_unwanted.append(i)
    elif label == 8:
        class8 += 1
        if class8 > 2500:
            list_of_unwanted.append(i)
    elif label == 9:
        class9 += 1
        if class9 > 2500:
            list_of_unwanted.append(i)
    i += 1
print(class0, class1, class2, class3, class4, class5, class6, class7, class8, class9)
print("sum:", class0 + class1 + class2 + class3 + class4 + class5 + class6 + class7 + class8 + class9)

print(list_of_unwanted)


# Did the same for 25% split. Checking if its correct now:
print("Checking the 25% split:")
t0 = time.process_time()    #Check duration of tfds.load (print is below tfds.load)
(train_data, test_data), info = tfds.load(name='cifar10',
                  data_dir='~\\tensorflow_datasets',
                  split=['train[:12043]+train[12044:12047]+train[12048:12056]+train[12057:12062]+train['
                         '12064:12067]+train[12068:12074]+train[12075:12077]+train[12078:12091]+train['
                         '12092:12096]+train[12097:12101]+train[12102:12105]+train[12106:12109]+train['
                         '12110:12114]+train[12115:12119]+train[12122:12135]+train[12136:12151]+train['
                         '12152:12155]+train[12156:12163]+train[12164:12174]+train[12176:12188]+train['
                         '12189:12217]+train[12219:12220]+train[12221:12225]+train[12226:12236]+train['
                         '12237:12253]+train[12254:12255]+train[12256:12257]+train[12258:12266]+train['
                         '12267:12269]+train[12270:12274]+train[12275:12280]+train[12281:12283]+train['
                         '12285:12293]+train[12294:12304]+train[12305:12320]+train[12321:12322]+train['
                         '12323:12328]+train[12329:12331]+train[12332:12336]+train[12337:12340]+train['
                         '12342:12344]+train[12345:12346]+train[12347:12359]+train[12360:12361]+train['
                         '12362:12366]+train[12368:12371]+train[12372:12378]+train[12380:12384]+train['
                         '12385:12386]+train[12387:12393]+train[12394:12401]+train[12402:12404]+train['
                         '12406:12409]+train[12410:12411]+train[12412:12413]+train[12414:12425]+train['
                         '12426:12429]+train[12430:12435]+train[12436:12437]+train[12438:12440]+train['
                         '12442:12449]+train[12451:12452]+train[12454:12457]+train[12458:12468]+train['
                         '12469:12477]+train[12480:12481]+train[12483:12485]+train[12486:12488]+train['
                         '12490:12492]+train[12498:12505]+train[12507:12508]+train[12510:12515]+train['
                         '12516:12523]+train[12524:12526]+train[12529:12530]+train[12532:12533]+train['
                         '12534:12536]+train[12538:12539]+train[12540:12542]+train[12543:12545]+train['
                         '12547:12549]+train[12551:12552]+train[12554:12555]+train[12556:12558]+train['
                         '12561:12562]+train[12566:12567]+train[12568:12569]+train[12571:12572]+train['
                         '12575:12578]+train[12581:12587]+train[12588:12589]+train[12590:12591]+train['
                         '12598:12599]+train[12603:12607]+train[12613:12615]+train[12616:12617]+train['
                         '12619:12620]+train[12622:12623]+train[12624:12625]+train[12626:12627]+train['
                         '12629:12630]+train[12631:12633]+train[12634:12636]+train[12641:12642]+train['
                         '12648:12649]+train[12652:12653]+train[12658:12659]+train[12666:12667]+train['
                         '12668:12669]+train[12675:12677]+train[12678:12679]+train[12682:12683]+train['
                         '12695:12696]+train[12698:12700]+train[12703:12704]+train[12706:12707]+train['
                         '12710:12711]+train[12719:12720]+train[12730:12731]+train[12737:12738]+train['
                         '12739:12740]+train[12751:12752]+train[12760:12761]+train[12771:12772]+train[12788:12789]',
                         'test'],
                  shuffle_files=False,
                  with_info=True)
t1 = time.process_time() - t0
print(f"Time elapsed: {t1}seconds (CPU seconds elapsed (floating point))")


def _map_data(*args):
    image = args[0]['image']
    label = args[0]['label']
    # label = tf.one_hot(label, info.features['label'].num_classes)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


ds_neu = train_data.map(map_func=_map_data)

class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0
class5 = 0
class6 = 0
class7 = 0
class8 = 0
class9 = 0

list_of_unwanted = []

i = 0

for image, label in ds_neu.take(50000):
    if label == 0:
        class0 += 1
        if class0 > 1250:
            list_of_unwanted.append(i)
    elif label == 1:
        class1 += 1
        if class1 > 1250:
            list_of_unwanted.append(i)
    elif label == 2:
        class2 += 1
        if class2 > 1250:
            list_of_unwanted.append(i)
    elif label == 3:
        class3 += 1
        if class3 > 1250:
            list_of_unwanted.append(i)
    elif label == 4:
        class4 += 1
        if class4 > 1250:
            list_of_unwanted.append(i)
    elif label == 5:
        class5 += 1
        if class5 > 1250:
            list_of_unwanted.append(i)
    elif label == 6:
        class6 += 1
        if class6 > 1250:
            list_of_unwanted.append(i)
    elif label == 7:
        class7 += 1
        if class7 > 1250:
            list_of_unwanted.append(i)
    elif label == 8:
        class8 += 1
        if class8 > 1250:
            list_of_unwanted.append(i)
    elif label == 9:
        class9 += 1
        if class9 > 1250:
            list_of_unwanted.append(i)
    i += 1
print(class0, class1, class2, class3, class4, class5, class6, class7, class8, class9)
print("sum:", class0 + class1 + class2 + class3 + class4 + class5 + class6 + class7 + class8 + class9)

print(list_of_unwanted)


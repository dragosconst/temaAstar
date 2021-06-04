import copy
import numpy as np
import os
import time
from multiprocessing import Process
from threading import Timer, Thread, Event
import sys

input_fpath = input("Unde e folderul pentru input?")
input_fname = "\\" + input("Cum se numeste fisierul de input?")
output_fpath = input("Unde e folderul pentru output?")
output_fname = "\\" + input("Cum se numeste fisierul de output?")
n_sol = int(input("Cat este NSOL?"))
timeout = float(input("Care este timpul de timeout?"))
mal_id = "mal"  # id rezervat pt mal

radius = 0
frogs = []  # o sa fie o lista de (id_broscoi, greutate_broscoi, id_frunza)
leaves = []  # o sa fie o lista de (id_frunza, xi, yi, nr_i, g_max_i)
try:
    file = open(input_fpath + input_fname, 'r')
    file_lines = file.read()
    for file_line in file_lines.splitlines():
        if file_line == file_lines.splitlines()[0]:
            radius = float(file_lines.splitlines()[0])
        else:
            if file_line == file_lines.splitlines()[1]:
                file_line = file_line.split()
                index = 0
                while index < len(file_line):
                    br_info = [file_line[index], int(file_line[index + 1]), file_line[index + 2]]
                    frogs += [br_info]
                    index += 3
            else:  # citim frunzele acum
                file_line = file_line.split()
                leaf = [file_line[0], float(file_line[1]), float(file_line[2]), int(file_line[3]), float(file_line[4])]
                leaves += [leaf]
except IOError as e:
    print(e.filename, e.strerror)

try:
    os.remove(output_fpath + output_fname)
except OSError as e:
    print(e.filename, e.strerror)
output = open(output_fpath + output_fname, "a")

print(radius)
print(frogs)
print(leaves)


# calculeaza distanta intre doua frunzem
# daca l2 == 'mal', calculeaza distanta de mal
# l1 e mereu doar id-ul
def distance_leaves(l1, l2):
    for leaf in leaves:
        if leaf[0] == l1:
            l1 = (leaf[1], leaf[2])
            break
    if l2 == 'mal':
        if radius - np.sqrt(l1[0] ** 2 + l1[1] ** 2) < 0:
            print(radius - np.sqrt(l1[0] ** 2 + l1[1] ** 2))
        return radius - np.sqrt(l1[0] ** 2 + l1[1] ** 2)

    for leaf in leaves:
        if leaf[0] == l2[0]:
            l2 = (leaf[1], leaf[2])
            break
    if np.sqrt((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2) < 0:
        print(np.sqrt((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2))
    return np.sqrt((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2)


# clasa utility ca sa apelez ce euristica vreau
class Euristici:
    EURISTICA_BANALA = 0
    EURISTICA_GRESITA = 1
    EURISTICA_BUNA_1 = 2
    EURISTICA_BUNA_2 = 3

    def __init__(self):
        pass

    @staticmethod
    def call_heuristic(gr, info, which):
        if which == Euristici.EURISTICA_BANALA:
            return gr.calculeaza_h_banal(info)
        elif which == Euristici.EURISTICA_GRESITA:
            return gr.calculeaza_h_wrong(info)
        elif which == Euristici.EURISTICA_BUNA_1:
            return gr.calculeaza_h1(info)
        else:
            return gr.calculeaza_h2(info)


class NodParcurgere:

    # format info = [Nume broasca, id frunza]
    # format info_extra = [gr_broaste, leaves], unde gr_broaste rep greutatile fiecarei broastes
    def __init__(self, info, info_extra, parinte, g, f):
        self.info = info
        self.info_extra = info_extra
        self.parinte = parinte  # parintele din arborele de parcurgere
        self.f = f
        self.g = g

    def obtineDrum(self):
        l = [(self.info, self.info_extra, self.f, self.g)]
        nodes = [self]
        nod = self
        while nod.parinte is not None:
            l.insert(0, (nod.parinte.info, nod.parinte.info_extra, nod.parinte.f, nod.parinte.g))
            nodes.insert(0, nod.parinte)
            nod = nod.parinte
        return l, nodes

    def afisDrum(self, g):  # returneaza si lungimea drumului
        l, nodes = self.obtineDrum()
        print(("->").join([str(x) for x in l]))
        print("mutari facute: " + str(len(l) - 1))
        return len(l)

    def contineInDrum(self, infoNodNou, infoExtraNodNoud):
        nodDrum = self
        while nodDrum is not None:
            if infoNodNou == nodDrum.info and infoExtraNodNoud == nodDrum.info_extra:
                return True
            nodDrum = nodDrum.parinte

        return False

    def __repr__(self):
        sir = ""
        sir += self.info + "("
        sir += "id = {}, ".format(self.id)
        sir += "drum="
        drum = self.obtineDrum()
        sir += ("->").join(drum)
        sir += " cost:{})".format(self.h)
        return (sir)


class Graph:  # graful problemei
    def __init__(self, start, start_extra, scopuri):
        self.start = start
        self.start_extra = start_extra
        self.end = scopuri

    def indiceNod(self, n):
        return self.noduri.index(n)

    # verifica daca e un state din care ar fi imposibil sa ajunga la mal, caz in care il ignora
    def bad_state(self, info, info_extra):
        frogs = info
        leaves = info_extra[-1]
        for index in range(len(frogs)):
            frog = frogs[index]
            weight = info_extra[index]
            # nu s sigur daca e necesar, dar verific cazul in care cumva o greutate a ajuns 0
            if weight == 0:
                return True
            cr_leaf = frog[1]

            if cr_leaf == mal_id:
                continue

            cantReachAny = True
            # ma uit daca ar putea ajunge oriunde, inclusiv pe mal
            for leaf in leaves:
                if leaf[0] == cr_leaf:
                    for insecte in range(leaf[3] + 1):
                        pos_w = weight + insecte
                        pos_tol = leaf[4] - insecte
                        if pos_tol < 0:
                            break
                        if distance_leaves(cr_leaf, mal_id) <= pos_w / 3:
                            cantReachAny = False
                            break
                        for other_leaf in leaves:
                            if other_leaf[0] == cr_leaf:
                                continue
                            if distance_leaves(cr_leaf, other_leaf) <= pos_w / 3 and \
                                other_leaf[4] >= pos_w - 1:
                                cantReachAny = False
                                break
                    break
            # daca exista macar o broasca care ramane blocata pe frunza actuala, inseamna
            # ca e o stare din care nu ar putea ajunge la mal sigur
            if cantReachAny:
                return True
        return False

    # e o functie recursiva, face un fel de backtracking ca sa genereze toate posibilitatile de unde ma aflu
    def generate_all_succ(self, succ, current, current_extra, g, frogs, info_extra, euristica):
        leaves = copy.deepcopy(info_extra[-1])
        # print(current, frogs, g)
        if frogs == []:
            if not self.bad_state(current, current_extra + [leaves]):
                succ += [[current, current_extra + [leaves], Euristici.call_heuristic(self, current, euristica), g]]
            # print(current, current_extra + [leaves])
            return
        frog = frogs[0]
        greutate = copy.deepcopy(info_extra[0])
        current_leaf = frog[1]
        # le pun si sa manance
        if current_leaf != mal_id:
            for leaf in leaves:
                if leaf[0] == current_leaf:
                    # incerc toate posibilitatile de insecte
                    for insecte in range(leaf[3] + 1):
                        greutate += (1 if insecte > 0 else 0)
                        # print(insecte, greutate, frog[0], frog[1])
                        leaf[3] -= (1 if insecte > 0 else 0)
                        leaf[4] -= (1 if insecte > 0 else 0)
                        if 0 > leaf[4]:
                            break
                        # salturile
                        for other_leaf in leaves:
                            if current_leaf != mal_id and other_leaf[0] != current_leaf and \
                                    distance_leaves(current_leaf, other_leaf) <= greutate / 3 and \
                                    other_leaf[4] >= (greutate - 1):
                                # print(leaf[0], current_leaf)
                                other_leaf[4] -= (greutate - 1)
                                leaf[4] += greutate
                                self.generate_all_succ(succ, current + [[frog[0], other_leaf[0]]],
                                                       current_extra + [greutate - 1],
                                                       g + distance_leaves(current_leaf, other_leaf),
                                                       frogs[1:], info_extra[1:-1] + [leaves],
                                                       euristica)
                                other_leaf[4] += (greutate - 1)
                                leaf[4] -= greutate
                        # print(greutate, frog[0], current_leaf, distance_leaves(current_leaf, mal_id))
                        # print(frog[0], current_leaf, distance_leaves(current_leaf, mal_id), "mal", greutate, greutate / 3, insecte, leaf[3])
                        if current_leaf != mal_id and distance_leaves(current_leaf, mal_id) <= greutate / 3:
                            leaf[4] += greutate
                            self.generate_all_succ(succ, current + [[frog[0], mal_id]],
                                                   current_extra + [greutate - 1],
                                                   g + distance_leaves(current_leaf, mal_id),
                                                   frogs[1:], info_extra[1:-1] + [leaves],
                                                   euristica)
                            leaf[4] -= greutate
                    break
        else:
            self.generate_all_succ(succ, current + [[frog[0], mal_id]],
                                   current_extra + [greutate - 1],
                                   g,
                                   frogs[1:], info_extra[1:-1] + [leaves],
                                   euristica)

    # va genera succesorii sub forma de noduri in arborele de parcurgere
    def genereazaSuccesori(self, nodCurent, euristica):
        succ = []
        self.generate_all_succ(succ, [], [], nodCurent.g, nodCurent.info, nodCurent.info_extra, euristica)
        return succ

    # distanta de la broaste catre mal
    def calculeaza_h1(self, info):
        h = 0.0
        for frog in info:
            if frog[1] == mal_id:
                continue
            h += distance_leaves(frog[1], mal_id)
        return h

    # distanta catre centrul cercului
    def calculeaza_h_wrong(self, info):
        h = 0.0
        for frog in info:
            for leaf in leaves:
                if leaf[0] == frog[1]:
                    h += np.sqrt(leaf[1] ** 2 + leaf[2] ** 2)
                    break
        return h

    # pentru ca costurile pot fi minim 0, cand toate broastele sunt la mal
    # altfel e mereu o valoare pozitiva
    def calculeaza_h_banal(self, info):
        return 0

    # minimul dintre distanta unei broaste catre mal si catre cea mai apropiata frunza
    def calculeaza_h2(self, info):
        h = 0.0
        for frog in info:
            if frog[1] == mal_id:
                continue
            dist_to_mal = distance_leaves(frog[1], mal_id)
            smallest_dist = None
            for leaf in leaves:
                if leaf[0] == frog[1]: # atentie sa nu calculez distanta cate aceeasi frunza
                    continue
                if smallest_dist is None or distance_leaves(frog[1], leaf) < smallest_dist:
                    smallest_dist = distance_leaves(frog[1], leaf)
            h += min(dist_to_mal, smallest_dist)
        return h

    def __repr__(self):
        sir = ""
        for (k, v) in self.__dict__.items():
            sir += "{} = {}\n".format(k, v)
        return sir


# Date de intrare
start = [[x[0], x[2]] for x in frogs]  # numele si frunza pe care se afla
start_extra = [x[1] for x in frogs]
# tine cont de greutatile broscoilor de pe fruzne
for leaf in leaves:
    for frog in frogs:
        if frog[2] == leaf[0]:
            leaf[4] -= frog[1]
start_extra += [leaves]
print(start, start_extra)
scopuri = [[x[0], mal_id] for x in frogs]
print("scopuri", scopuri)

gr = Graph(start, start_extra, scopuri)


def in_list(nod_info, nod_info_extra, lista):
    for nod in lista:
        if nod_info == nod.info and nod_info_extra == nod.info_extra:
            return nod
    return None


def insert(node, lista):
    idx = 0
    while idx < len(lista) and (node.f > lista[idx].f or (node.f == lista[idx].f and node.g < lista[idx].g)):
        idx += 1
    lista.insert(idx, node)


def write_output(last_node, nodes_in_mem, max_nodes_in_mem, my_start_time, file):
    drum, nodes = last_node.obtineDrum()
    node_ind = 0
    old_weight = [0 for i in range(len(start))]
    old_leaf = ["" for i in range(len(start))]

    file.write("Timp total executie: " + str(time.time() - my_start_time) + "\n")

    for (info, info_extra, f, g) in drum:
        file.write(str(node_ind) + ")\n")
        f_index = 0
        for frog in info:
            if old_leaf[f_index] == mal_id:
                continue
            if info == start and info_extra == start_extra:
                file.write(frog[0] + " se afla pe frunza initiala " + frog[1] + ". Greutate broscuta " + str(
                    info_extra[f_index]) + "\n")
            else:
                file.write(frog[0] + " a mancat " + str(info_extra[f_index] + 1 - old_weight[f_index]) + " insecte. " + \
                           frog[0] + " a sarit de la " + old_leaf[f_index] + " la " + frog[
                               1] + ". Greutate broscuta " + str(info_extra[f_index]) + "\n")

            old_weight[f_index] = info_extra[f_index]
            old_leaf[f_index] = frog[1]
            f_index += 1
        file.write("Stare frunze: " + str(info_extra[-1]) + "\n")
        node_ind += 1
    file.write("Cost total: " + str(drum[-1][2]) + "\n")
    file.write("Noduri folosite de algoritm in total: " + str(nodes_in_mem) + "\n")
    file.write("Noduri folosite maxim in memorie la un moment dat de algoritm: " + str(max_nodes_in_mem) + "\n")
    file.write("-"*150 + "\n")
    file.write("-"*150 + "\n")
    file.flush()

def write_message(msg, file):
    file.write(msg)
    file.flush()


def a_star_optim(euristica):
    global n_sol, timeout

    nrSolutiiCautate = n_sol
    my_start_time = time.time()
    open = [NodParcurgere(start, start_extra, None, 0, Euristici.call_heuristic(gr, start, euristica))]
    closed = []

    while len(open) > 0:
        current = open.pop(0)
        closed.append(current)
        if current.info == scopuri:
            if time.time() - my_start_time > timeout:
                sys.exit()
            write_output(current, len(open) + len(closed), len(open) + len(closed), my_start_time, output)
            nrSolutiiCautate -= 1
        if nrSolutiiCautate == 0:
            break

        succ = gr.genereazaSuccesori(current, euristica)
        if time.time() - my_start_time > timeout:
            sys.exit()
        for i in succ:
            info, info_extra, h, g = i
            if current.contineInDrum(info, info_extra):
                continue

            node = in_list(info, info_extra, closed)
            if node is not None:
                if g + h < node.f:
                    closed.remove(node)
                    insert(NodParcurgere(info, info_extra, current, g, g + h), open)
                continue

            node = in_list(info, info_extra, open)
            if node is not None:
                if g + h < node.f:
                    open.remove(node)
                    insert(NodParcurgere(info, info_extra, current, g, g + h), open)
                continue

            insert(NodParcurgere(info, info_extra, current, g, g + h), open)

def a_star(euristica):
    global n_sol, timeout

    nrSolutiiCautate = n_sol
    my_start_time = time.time()

    continua = True
    c = [NodParcurgere(start, start_extra, None, 0, Euristici.call_heuristic(gr, start, euristica))]
    pasi = 0

    while len(c) > 0 and continua:

        nod = c.pop(0)
        pasi += 1
        if nod.info == scopuri:
            if time.time() - my_start_time > timeout:
                sys.exit()
            write_output(nod, len(c) + pasi, len(c) + pasi, my_start_time, output)
            nrSolutiiCautate = nrSolutiiCautate - 1
        if nrSolutiiCautate == 0:
            continua = False
            continue

        succ = gr.genereazaSuccesori(nod, euristica)
        if time.time() - my_start_time > timeout:
            sys.exit()
        for (nodInfo, nodInfoExtra, h, g) in succ:
            if nod.contineInDrum(nodInfo, nodInfoExtra):
                continue
            newNod = NodParcurgere(nodInfo, nodInfoExtra, nod, g, h + g)
            insert(newNod, c)

def construieste_drum(nodCurent: NodParcurgere, nodes_so_far, mem_max, limita, euristica, my_start_time):
    global timeout

    if nodCurent.f > limita:
        return (False, nodCurent.f, nodes_so_far)
    if nodCurent.info == scopuri:
        if time.time() - my_start_time > timeout:
            sys.exit()
        write_output(nodCurent, nodes_so_far, max(mem_max, nodes_so_far), my_start_time, output)
        return (True, nodCurent.f, nodes_so_far)
    succ = gr.genereazaSuccesori(nodCurent, euristica)
    if time.time() - my_start_time > timeout:
        sys.exit()
    minif = float('inf')
    for (info, info_extra, h, g) in succ:
        ajuns, f, nodes_so_far = construieste_drum(NodParcurgere(info, info_extra, nodCurent, g, g + h), nodes_so_far + 1,
                                                   mem_max, limita, euristica, my_start_time)
        if ajuns:
            return (True,f, nodes_so_far)
        minif = min(minif, f)
    return (False, minif, nodes_so_far)


def ida_star(euristica):
    global timeout, n_sol

    my_start_time = time.time()
    nrSolutiiCautate = n_sol

    st = NodParcurgere(start, start_extra, None, 0, Euristici.call_heuristic(gr, start, euristica))
    limita = 0
    total_nodes = 0
    max_at_point = 0
    while True:
        ajuns, limita, current_nodes = construieste_drum(st, 1, max_at_point, limita, euristica, my_start_time)
        if time.time() - my_start_time > timeout:
            sys.exit()
        total_nodes += current_nodes
        max_at_point = max(max_at_point, current_nodes)
        if ajuns:
            nrSolutiiCautate -= 1
            if nrSolutiiCautate == 0:
                break
        if limita == float('inf'):
            print('Nu exista drum!')
            break


def uniform_cost(gr):
    queue = [NodParcurgere(start, start_extra, None, 0, 0)]
    my_start_time = time.time()
    pasi = 0
    while len(queue) > 0:
        curr = queue[0]
        queue.pop(0)
        pasi += 1
        if curr.info == scopuri:
            if time.time() - my_start_time > timeout:
                sys.exit()
            write_output(curr, len(queue) + pasi, len(queue) + pasi, my_start_time, output)
            break
        listasuccesori = gr.genereazaSuccesori(curr, Euristici.EURISTICA_BANALA)
        if time.time() - my_start_time > timeout:
            sys.exit()
        for (info, info_extra, h, f) in listasuccesori:
            poz = 0
            while poz < len(queue) and f > queue[poz].f:
                poz += 1
            queue.insert(poz, NodParcurgere(info, info_extra, curr, f, f))

if __name__ == "__main__":
    timeout += 5 # timpi rezervati pentru scrieri in fisier
    if gr.bad_state(start, start_extra):
        write_message("Nu se poate ajunge nicaieri din starea de inceput!\n", output)
    else:
        write_message("Rezultate cu A*:\n", output)
        for euristica in range(4):
            if euristica == Euristici.EURISTICA_BANALA:
                write_message("Cu euristica banala.\n", output)
            elif euristica == Euristici.EURISTICA_GRESITA:
                write_message("Cu euristica gresita.\n", output)
            elif euristica == Euristici.EURISTICA_BUNA_1:
                write_message("Cu euristica care tine cont de distanta catre mal.\n", output)
            else:
                write_message("Cu euristica care tine cont de cea mai apropiata frunza (sau mal).\n", output)

            start_time = time.time()
            print(start_time, timeout)
            t = Thread(target=a_star, args=(euristica,))
            t.start()
            t.join(timeout) # + 20 pt scrieri
            print(time.time() - start_time, timeout)
            if time.time() - start_time >= timeout:
                write_message("A fost depasit timeout-ul!\n", output)
                write_message("-" * 150 + "\n", output)

        write_message("-"*150 + "\n", output)
        write_message("Rezultate cu A* optim:\n", output)
        for euristica in range(4):
            if euristica == Euristici.EURISTICA_BANALA:
                write_message("Cu euristica banala.\n", output)
            elif euristica == Euristici.EURISTICA_GRESITA:
                write_message("Cu euristica gresita.\n", output)
            elif euristica == Euristici.EURISTICA_BUNA_1:
                write_message("Cu euristica care tine cont de distanta catre mal.\n", output)
            else:
                write_message("Cu euristica care tine cont de cea mai apropiata frunza (sau mal).\n", output)

            start_time = time.time()
            t = Thread(target=a_star_optim, args=(euristica,))
            t.start()
            t.join(timeout)
            if time.time() - start_time >= timeout:
                write_message("A fost depasit timeout-ul!\n", output)
                write_message("-" * 150 + "\n", output)

        write_message("-"*150 + "\n", output)
        write_message("Rezultate cu IDA*:\n", output)
        for euristica in range(4):
            if euristica == Euristici.EURISTICA_BANALA:
                write_message("Cu euristica banala.\n", output)
            elif euristica == Euristici.EURISTICA_GRESITA:
                write_message("Cu euristica gresita.\n", output)
            elif euristica == Euristici.EURISTICA_BUNA_1:
                write_message("Cu euristica care tine cont de distanta catre mal.\n", output)
            else:
                write_message("Cu euristica care tine cont de cea mai apropiata frunza (sau mal).\n", output)

            start_time = time.time()
            t = Thread(target=ida_star, args=(euristica,))
            t.start()
            t.join(timeout)
            if time.time() - start_time >= timeout:
                write_message("A fost depasit timeout-ul!\n", output)
                write_message("-" * 150 + "\n", output)

        write_message("-"*150 + "\n", output)
        write_message("Rezultate cu UCS:\n", output)
        start_time = time.time()
        t = Thread(target=uniform_cost, args=(gr,))
        t.start()
        t.join(timeout)
        if time.time() - start_time >= timeout:
            write_message("A fost depasit timeout-ul!\n", output)
            write_message("-"*150 + "\n", output)

import copy
import numpy as np
import os
import time

input_path = input("Unde e folderul de input?")
input_fname = "\\broscoi.input"  # e hard coded???
output_path = input("Unde e folderul de output?")
output_fname = "\\broscoi.output"
n_sol = int(input("Cate solutii sunt cautate?"))
timeout = float(input("Care este timpul de timeout?"))
mal_id = "mal"  # id rezervat pt mal

radius = 0
broscoi = []  # o sa fie o lista de (id_broscoi, greutate_broscoi, id_frunza)
leaves = []  # o sa fie o lista de (id_frunza, xi, yi, nr_i, g_max_i)
try:
    file = open(input_path + input_fname, 'r')
    contents = file.read()
    for line in contents.splitlines():
        if line == contents.splitlines()[0]:
            radius = float(contents.splitlines()[0])
        else:
            if line == contents.splitlines()[1]:
                line = line.split()
                index = 0
                while index < len(line):
                    br_info = [line[index], int(line[index + 1]), line[index + 2]]
                    broscoi += [br_info]
                    index += 3
            else:  # citim frunzele acum
                line = line.split()
                leaf = [line[0], float(line[1]), float(line[2]), int(line[3]), float(line[4])]
                leaves += [leaf]
except IOError as e:
    print(e.filename, e.strerror)

try:
    os.remove(output_path + output_fname)
except OSError as e:
    pass
output_file = open(output_path + output_fname, "a")

print(radius)
print(broscoi)
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


# clasa utility ca sa apelez cu un wrapper ce euristica vreau
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
            return gr.calculeaza_h(info)
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

    # e o functie recursiva, face un fel de backtracking ca sa genereze toate posibilitatile de unde ma aflu
    def generate_all_succ(self, succ, current, current_extra, g, frogs, info_extra, euristica):
        leaves = copy.deepcopy(info_extra[-1])
        # print(current, frogs, g)
        if frogs == []:
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

    # va genera succesorii sub forma de noduri in arborele de parcurgere
    def genereazaSuccesori(self, nodCurent, euristica):
        succ = []
        self.generate_all_succ(succ, [], [], nodCurent.g, nodCurent.info, nodCurent.info_extra, euristica)
        return succ

    # distanta de la broaste catre mal
    def calculeaza_h(self, info):
        h = 0.0
        for frog in info:
            if frog[1] == mal_id:
                continue
            h += distance_leaves(frog[1], mal_id)
        return h

    # euristica buna 1 + un nr foarte mare
    def calculeaza_h_wrong(self, info):
        h = self.calculeaza_h(info)
        h = 2 ** 10 - h
        return h

    # 0 e mereu acceptabil, pentru ca costurile pot fi minim 0, cand toate broastele sunt la mal
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
start = [[x[0], x[2]] for x in broscoi]  # numele si frunza pe care se afla
start_extra = [x[1] for x in broscoi]
# tine cont de greutatile broscoilor de pe fruzne
for leaf in leaves:
    for frog in broscoi:
        if frog[2] == leaf[0]:
            leaf[4] -= frog[1]
start_extra += [leaves]
print(start, start_extra)
scopuri = [[x[0], mal_id] for x in broscoi]
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


def write_output(last_node, all_nodes, max_mem_nodes, file):
    global start_time

    drum, nodes = last_node.obtineDrum()
    index = 0
    old_weight = [0 for i in range(len(start))]
    old_leaf = ["" for i in range(len(start))]

    file.write("Timp total executie: " + str(time.time() - start_time) + "\n")

    for (info, info_extra, f, g) in drum:
        file.write(str(index) + ")\n")
        f_index = 0
        for frog in info:
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
        index += 1
    file.write("Cost total: " + str(drum[-1][2]) + "\n")
    file.write("Noduri folosite in total: " + str(all_nodes) + "\n")
    file.write("Noduri folosite maxim in memorie la un moment dat: " + str(max_mem_nodes) + "\n")
    file.write("-"*50 + "\n\n")
    file.flush()

def write_message(msg, file):
    file.write(msg)
    file.flush()

def a_star_optim():
    global n_sol, start_time, timeout

    for euristica in range(4):
        write_message("Cu euristica " + ("banala" if euristica == Euristici.EURISTICA_BANALA
                                         else "gresita" if euristica == Euristici.EURISTICA_GRESITA
                                         else "h1" if euristica == Euristici.EURISTICA_BUNA_1
                                         else "h2" if euristica == Euristici.EURISTICA_BUNA_2 else "none")
                                        + ".\n", output_file)
        nrSolutiiCautate = n_sol
        start_time = time.time()
        open = [NodParcurgere(start, start_extra, None, 0, Euristici.call_heuristic(gr, start, euristica))]
        closed = []

        while len(open) > 0:
            # testez timeout, nu e cea mai exacta metoda
            if time.time() - start_time > timeout:
                write_message("A* optim a depasit timeout\n", output_file)
                break
            current = open.pop(0)
            closed.append(current)
            if current.info == scopuri:
                write_output(current, len(open) + len(closed), len(open) + len(closed), output_file)
                nrSolutiiCautate -= 1
            if nrSolutiiCautate == 0:
                break

            # testez timeout, nu e cea mai exacta metoda
            if time.time() - start_time > timeout:
                write_message("A* optim a depasit timeout\n", output_file)
                break
            succ = gr.genereazaSuccesori(current, euristica)
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

                # testez timeout, nu e cea mai exacta metoda
                if time.time() - start_time > timeout:
                    write_message("A* optim a depasit timeout\n", output_file)
                    break

                node = in_list(info, info_extra, open)
                if node is not None:
                    if g + h < node.f:
                        open.remove(node)
                        insert(NodParcurgere(info, info_extra, current, g, g + h), open)
                    continue

                insert(NodParcurgere(info, info_extra, current, g, g + h), open)

def a_star():
    global n_sol, start_time, timeout

    for euristica in range(4):
        write_message("Cu euristica " + ("banala" if euristica == Euristici.EURISTICA_BANALA
                                         else "gresita" if euristica == Euristici.EURISTICA_GRESITA
                                         else "h1" if euristica == Euristici.EURISTICA_BUNA_1
                                         else "h2" if euristica == Euristici.EURISTICA_BUNA_2 else "none")
                                        + ".\n", output_file)
        nrSolutiiCautate = n_sol
        start_time = time.time()

        continua = True
        c = [NodParcurgere(start, start_extra, None, 0, Euristici.call_heuristic(gr, start, euristica))]
        pasi = 0

        while len(c) > 0 and continua:
            # testez timeout, nu e cea mai exacta metoda
            if time.time() - start_time > timeout:
                write_message("A* a depasit timeout\n", output_file)
                break

            nod = c.pop(0)
            pasi += 1
            if nod.info == scopuri:
                write_output(nod, len(c) + pasi, len(c) + pasi, output_file)
                nrSolutiiCautate = nrSolutiiCautate - 1
            if nrSolutiiCautate == 0:
                continua = False

            succ = gr.genereazaSuccesori(nod, euristica)
            # testez timeout, nu e cea mai exacta metoda
            if time.time() - start_time > timeout:
                write_message("A* a depasit timeout\n", output_file)
                break
            for (nodInfo, nodInfoExtra, h, g) in succ:
                if nod.contineInDrum(nodInfo, nodInfoExtra):
                    continue

                # testez timeout, nu e cea mai exacta metoda
                if time.time() - start_time > timeout:
                    write_message("A* a depasit timeout\n", output_file)
                    break
                newNod = NodParcurgere(nodInfo, nodInfoExtra, nod, g, h + g)
                insert(newNod, c)

def construieste_drum(nodCurent: NodParcurgere, nodes_so_far, mem_max, limita):
    global start_time, timeout

    if nodCurent.f > limita:
        return (False, nodCurent.f, nodes_so_far)
    if nodCurent.info == scopuri:
        write_output(nodCurent, nodes_so_far, max(mem_max, nodes_so_far), output_file)
        return (True, nodCurent.f, nodes_so_far)
    succ = gr.genereazaSuccesori(nodCurent)
    minif = float('inf')
    for (info, info_extra, h, g) in succ:
        # testez timeout, nu e cea mai exacta metoda
        if time.time() - start_time > timeout:
            write_message("IDA* a depasit timeout\n", output_file)
            break
        ajuns, f, nodes_so_far = construieste_drum(NodParcurgere(info, info_extra, nodCurent, g, g + h), nodes_so_far + 1,
                                                   mem_max, limita)
        if ajuns:
            return (True,f, nodes_so_far)
        minif = min(minif, f)
    return (False, minif, nodes_so_far)


def ida_star():
    global n_sol, start_time, timeout

    nrSolutiiCautate = n_sol
    start_time = time.time()

    st = NodParcurgere(start, start_extra, None, 0, Euristici.call_heuristic(gr, start, 2))
    limita = 0
    all_nodes = 0
    max_mem = 0
    while True:
        ajuns, limita, gen_nodes = construieste_drum(st, 1, max_mem, limita)
        # testez timeout, nu e cea mai exacta metoda
        if time.time() - start_time > timeout:
            write_message("IDA* a depasit timeout\n", output_file)
            break
        all_nodes += gen_nodes
        max_mem = max(max_mem, gen_nodes)
        if ajuns:
            nrSolutiiCautate -= 1
            if nrSolutiiCautate == 0:
                break
        if limita == float('inf'):
            print('Nu exista drum!')
            break

write_message("Rezultate cu A*:\n", output_file)
start_time = time.time()
print(start_time, timeout)
a_star()
write_message("Rezultate cu A* optim:\n", output_file)
a_star_optim()
write_message("Rezultate cu IDA*:\n", output_file)
ida_star()

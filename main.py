import copy
import numpy as np

input_path  = input("Unde e folderul de input?")
input_fname   = "\\broscoi.input" # e hard coded???
output_path = input("Unde e folderul de output?")
output_fname = "\\broscoi.output"
n_sol       = int(input("Cate solutii sunt cautate?"))
timeout     = float(input("Care este timpul de timeout?"))
mal_id      = "mal" # id rezervat pt mal

radius = 0
broscoi = [] # o sa fie o lista de (id_broscoi, greutate_broscoi, id_frunza)
frunze = [] # o sa fie o lista de (id_frunza, xi, yi, nr_i, g_max_i)
max_weight = -1
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
                    if int(line[index + 1]) > max_weight:
                        max_weight = int(line[index + 1])
                    broscoi += [br_info]
                    index += 3
            else: # citim frunzele acum
                line = line.split()
                frunza = [line[0], float(line[1]), float(line[2]), int(line[3]), float(line[4])]
                max_weight += int(line[3])
                frunze += [frunza]
except IOError as e:
    print(e.filename, e.strerror)

print(radius)
print(broscoi)
print(frunze)

# calculeaza distanta intre doua frunzem
# daca l2 == 'mal', calculeaza distanta de mal
# l1 e mereu doar id-ul
def distance_leaves(l1, l2):
    for leaf in frunze:
        if leaf[0] == l1:
            l1 = (leaf[1], leaf[2])
            break
    if l2 == 'mal':
        return radius - np.sqrt(l1[0]**2 + l1[1] **2)

    for leaf in frunze:
        if leaf[0] == l2[0]:
            l2 = (leaf[1], leaf[2])
            break
    return np.sqrt((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2)

class NodParcurgere:

    # info e o lista cu structura asemanatoare cu lista de input de broscoi
    def __init__(self, info, leaves, parinte, g, f):
        self.info = info
        self.parinte = parinte  # parintele din arborele de parcurgere
        self.leaves = leaves
        self.f = f
        self.g = g

    def obtineDrum(self):
        l = [self.info]
        nodes = [self]
        nod = self
        while nod.parinte is not None:
            l.insert(0, (nod.parinte.info, nod.parinte.f, nod.parinte.g))
            nodes.insert(0, nod.parinte)
            nod = nod.parinte
        return l, nodes

    def afisDrum(self, g):  # returneaza si lungimea drumului
        l, nodes = self.obtineDrum()
        print(("->").join([str(x) for x in l]))
        print("mutari facute: " + str(len(l) - 1))
        return len(l)

    def contineInDrum(self, infoNodNou):
        nodDrum = self
        while nodDrum is not None:
            if (infoNodNou == nodDrum.info):
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
    def __init__(self, start, scopuri):
        self.start = start
        self.end = scopuri

    def indiceNod(self, n):
        return self.noduri.index(n)



    # OK ce ramane de facut:
    #TODO: tine minte cumva in stari si toate frunzele, ca n ai altfel cum sa scapi de situatia cand \
    #mananca insecte si trebuie sa ai memorie la faptul ca au mancat
    def generate_all_succ(self, succ, current, g, frogs):
        leaves = copy.deepcopy(frunze) # TODO: rename frunze to leaves
        # print(current, frogs, g)
        if frogs == []:
            succ += [[current, self.calculeaza_h(current), g]]
            return
        frog = frogs[0]
        greutate = frog[1]
        current_leaf = frog[2]
        # le pun si sa manance
        if current_leaf != mal_id:
            for leaf in leaves:
                if leaf[0] == current_leaf:
                    # incerc toate posibilitatile de insecte
                    for insecte in range(leaf[3] + 1):
                        if greutate + insecte > leaf[4]:
                            break
                        greutate += insecte
                        leaf[3] -= insecte
                        leaf[4] -= greutate
                        # salturile
                        for other_leaf in leaves:
                            if current_leaf != mal_id and other_leaf[0] != current_leaf and \
                                    distance_leaves(current_leaf, other_leaf) <= greutate / 3 and \
                                    other_leaf[4] >= (greutate - 1):
                                # print(leaf[0], current_leaf)
                                other_leaf[4] -= greutate - 1
                                self.generate_all_succ(succ, current + [[frog[0], greutate - 1, other_leaf[0]]],
                                                       g + distance_leaves(current_leaf, other_leaf), frogs[1:])
                                other_leaf[4] += greutate - 1
                        if current_leaf != mal_id and distance_leaves(current_leaf, mal_id) <= greutate / 3:
                            self.generate_all_succ(succ, current + [[frog[0], greutate - 1, mal_id]], g + distance_leaves(current_leaf, mal_id),
                                                   frogs[1:])
                        leaf[3] += insecte
                        leaf[4] += greutate
                        greutate -= insecte
                    break

    # va genera succesorii sub forma de noduri in arborele de parcurgere
    def genereazaSuccesori(self, nodCurent):
        # de modificat
        succ = []
        self.generate_all_succ(succ, [], nodCurent.g, nodCurent.info)
        return succ

    # distanta de la broaste catre mal
    def calculeaza_h(self, frogs):
        h = 0.0
        for frog in frogs:
            if frog[2] == mal_id:
                continue
            h += distance_leaves(frog[2], mal_id)
        return h

    def __repr__(self):
        sir = ""
        for (k, v) in self.__dict__.items():
            sir += "{} = {}\n".format(k, v)
        return sir

# toate combinatiile posibile in care pot ajunge la mal, tinand cont de numarul total de insecte + greutatea maxima
def build_final_states(scopuri, curent, broscoi_left, maxw):
    if len(broscoi_left) == 0:
        scopuri += [curent]
        return
    for posw in range(1, maxw + 1):
        build_final_states(scopuri, curent + [[broscoi_left[0][0], posw, mal_id]], broscoi_left[1:], maxw - posw)

# Date de intrare
start = broscoi
# print(max_weight)
scopuri = []
build_final_states(scopuri, [], broscoi, int(max_weight))
print(scopuri)


gr = Graph(start, scopuri)


def in_list(nod_info, lista):
    for nod in lista:
        if nod_info == nod.info:
            return nod
    return None


def insert(node, lista):
    idx = 0
    while idx < len(lista) and (node.f > lista[idx].f or (node.f == lista[idx].f and node.g < lista[idx].g)):
        idx += 1
    lista.insert(idx, node)

def a_star_optim():
  global n_sol
  open=[NodParcurgere(start, frunze, None, 0, gr.calculeaza_h(start))]
  closed=[]

  while len(open) > 0:
    current = open.pop(0)
    closed.append(current)
    if current.info in scopuri:
        current.afisDrum(gr)
        n_sol -= 1
    if n_sol == 0:
        break


    succ = gr.genereazaSuccesori(current)
    for i in succ:
      info, h, g = i
      # print(info)
      if current.contineInDrum(info):
        continue

      node = in_list(info, closed)
      if node is not None:
        if g + h < node.f:
          closed.remove(node)
          insert(NodParcurgere(info, frunze, current, g, g + h), open)
        continue

      node = in_list(info, open)
      if node is not None:
        if g + h < node.f:
          open.remove(node)
          insert(NodParcurgere(info, frunze, current, g, g + h), open)
        continue

      insert(NodParcurgere(info, frunze, current, g, g + h), open)


a_star_optim()
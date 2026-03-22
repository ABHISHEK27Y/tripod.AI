"""quant.py — Kalman Filter + A* + Euclidean Distance · Hikari Shinro AI"""
import numpy as np
import heapq
from scipy.linalg import inv
from typing import Dict, List, Tuple, Optional


class KalmanTracker:
    _id = 0

    def __init__(self, cx, cy):
        KalmanTracker._id += 1
        self.id      = KalmanTracker._id
        self.missed  = 0
        self.history = []
        dt = 1.0
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 5.0
        self.x = np.array([[cx],[cy],[0.],[0.]], dtype=float)
        self.P = np.eye(4) * 500.

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.history.append((float(self.x[0]), float(self.x[1])))
        if len(self.history) > 20: self.history.pop(0)
        return float(self.x[0]), float(self.x[1])

    def update(self, cx, cy):
        z = np.array([[cx],[cy]], dtype=float)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.missed = 0


class MultiObjectTracker:
    def __init__(self):
        self.trackers: Dict[str, KalmanTracker] = {}

    def update(self, detections):
        active = set()
        for d in detections:
            lbl = d["label"]; cx, cy = d["cx"], d["cy"]
            active.add(lbl)
            if lbl not in self.trackers:
                self.trackers[lbl] = KalmanTracker(cx, cy)
            else:
                self.trackers[lbl].predict()
                self.trackers[lbl].update(cx, cy)
            d["tracker_id"] = self.trackers[lbl].id
            d["trail"]      = list(self.trackers[lbl].history[-10:])
        for lbl in list(self.trackers):
            if lbl not in active:
                self.trackers[lbl].missed += 1
                if self.trackers[lbl].missed > 8:
                    del self.trackers[lbl]
        return detections


class AStarNavigator:
    GW=32; GH=24; THRESH=0.72; RAD=1

    def build_grid(self, depth_map):
        from PIL import Image
        h,w = depth_map.shape[:2]
        dn  = depth_map.astype(float)
        if dn.max() > 0: dn /= dn.max()
        p   = Image.fromarray((dn*255).astype(np.uint8)).resize((self.GW,self.GH), Image.NEAREST)
        g   = np.array(p)/255. > self.THRESH
        exp = g.copy()
        for dy in range(-self.RAD, self.RAD+1):
            for dx in range(-self.RAD, self.RAD+1):
                exp |= np.roll(np.roll(g,dy,0),dx,1)
        return exp

    def find_path(self, depth_map, tx, ty):
        grid  = self.build_grid(depth_map)
        start = (self.GW//2, self.GH-1)
        goal  = (min(int(tx*self.GW), self.GW-1), min(int(ty*self.GH), self.GH-1))
        if grid[goal[1],goal[0]]:
            for r in range(1,5):
                found = False
                for dc in range(-r,r+1):
                    for dr in range(-r,r+1):
                        nc,nr = goal[0]+dc, goal[1]+dr
                        if 0<=nc<self.GW and 0<=nr<self.GH and not grid[nr,nc]:
                            goal=(nc,nr); found=True; break
                    if found: break
                if found: break
        heap = [(0, start)]; came = {start:None}; g_cost = {start:0.}
        h = lambda a,b: abs(a[0]-b[0])+abs(a[1]-b[1])
        dirs = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        while heap:
            _,cur = heapq.heappop(heap)
            if cur==goal:
                path=[]; c=cur
                while c: path.append(c); c=came[c]
                return list(reversed(path))
            for dc,dr in dirs:
                nb=(cur[0]+dc,cur[1]+dr)
                if not(0<=nb[0]<self.GW and 0<=nb[1]<self.GH): continue
                if grid[nb[1],nb[0]]: continue
                ng=g_cost[cur]+(1.414 if dc and dr else 1.)
                if nb not in g_cost or ng<g_cost[nb]:
                    g_cost[nb]=ng; heapq.heappush(heap,(ng+h(nb,goal),nb)); came[nb]=cur
        return None

    def path_to_text(self, path):
        if not path or len(path)<2: return "path clear"
        s,e=path[0],path[-1]; dx=e[0]-s[0]
        if abs(dx)<2: return "straight ahead"
        return "to your left" if dx<0 else "to your right"


class DistanceEstimator:
    SCALE = 1.2

    def estimate(self, depth_map, x1,y1,x2,y2):
        h,w=depth_map.shape[:2]
        x1c,y1c=max(0,x1),max(0,y1)
        x2c,y2c=min(x2,w),min(y2,h)
        region=depth_map[y1c:y2c,x1c:x2c].astype(float)
        if region.size==0 or depth_map.max()==0: return 99.
        rn=region/depth_map.max()
        f=rn.flatten(); f.sort()
        inner=f[len(f)//4:3*len(f)//4]
        mean=float(inner.mean()) if len(inner) else float(f.mean())
        if mean<1e-4: return 99.
        return round(min(self.SCALE/mean,15.),2)

    def to_text(self, d):
        if d<0.5: return "very close — stop!"
        if d<1.0: return f"{d:.1f}m — caution"
        if d<3.0: return f"{d:.1f}m ahead"
        return f"{d:.0f}m away"


CONF_THRESH = 0.40

def filter_detections(dets):
    passed = [d for d in dets if d.get("conf",0) >= CONF_THRESH]
    best = {}
    for d in passed:
        if d["label"] not in best or d["conf"] > best[d["label"]]["conf"]:
            best[d["label"]] = d
    return list(best.values())


def build_scene(detections, path_text):
    if not detections: return f"No objects detected. {path_text}."
    parts=[]
    for d in sorted(detections, key=lambda x: x.get("distance_m",99)):
        cx=d.get("cx_norm",0.5)
        side="left" if cx<0.35 else "right" if cx>0.65 else "centre"
        parts.append(f"{d['label']} at {d.get('distance_m','?')}m ({side})")
    return f"Objects: {', '.join(parts)}. Navigation: {path_text}."


class QuantEngine:
    def __init__(self):
        self.tracker   = MultiObjectTracker()
        self.navigator = AStarNavigator()
        self.estimator = DistanceEstimator()

    def process(self, raw, depth_map, target_label="", fw=640, fh=480):
        filtered = filter_detections(raw)
        for d in filtered:
            if depth_map is not None:
                d["distance_m"]    = self.estimator.estimate(depth_map, d["x1"],d["y1"],d["x2"],d["y2"])
                d["distance_text"] = self.estimator.to_text(d["distance_m"])
            else:
                d["distance_m"]=99.; d["distance_text"]="unknown"
            d["cx_norm"] = d["cx"]/fw if fw>0 else 0.5
            d["cy_norm"] = d["cy"]/fh if fh>0 else 0.5

        enriched = self.tracker.update(filtered)
        path=None; path_text="path clear"
        if depth_map is not None:
            tgt = next((d for d in enriched if d["label"]==target_label), None)
            if not tgt and enriched:
                tgt = min(enriched, key=lambda d: d.get("distance_m",99))
            if tgt:
                path      = self.navigator.find_path(depth_map, tgt["cx_norm"], tgt["cy_norm"])
                path_text = self.navigator.path_to_text(path)

        scene = build_scene(enriched, path_text)
        return enriched, scene, path

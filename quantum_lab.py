import pygame
import sys
import threading
import numpy as np
import matplotlib
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
WIDTH, HEIGHT = 1200, 800
UI_WIDTH = 300
SIM_WIDTH = WIDTH - UI_WIDTH
SIM_HEIGHT = 550
FPS = 60
WHITE = (255,255,255)
GRAY = (220,220,220)
DARK = (40,40,40)
BLUE = (60,120,255)
GREEN = (60,200,120)
pygame.font.init()
FONT = pygame.font.SysFont("Arial",14)
FONT_LG = pygame.font.SysFont("Arial",18,bold=True)
class Slider:
    def __init__(self,x,y,w,min_val,max_val,value,label):
        self.rect=pygame.Rect(x,y,w,6)
        self.min=min_val
        self.max=max_val
        self.value=value
        self.label=label
        self.dragging=False
    def draw(self,screen):
        pygame.draw.rect(screen,GRAY,self.rect)
        t=(self.value-self.min)/(self.max-self.min)
        hx=self.rect.x+t*self.rect.width
        pygame.draw.circle(screen,BLUE,(int(hx),self.rect.centery),8)
        txt=FONT.render(f"{self.label}: {self.value:.2f}",True,DARK)
        screen.blit(txt,(self.rect.x,self.rect.y-20))
    def handle(self,event):
        if event.type==pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging=True
        if event.type==pygame.MOUSEBUTTONUP:
            self.dragging=False
        if event.type==pygame.MOUSEMOTION and self.dragging:
            rel=max(0,min(event.pos[0]-self.rect.x,self.rect.width))
            self.value=self.min+(rel/self.rect.width)*(self.max-self.min)
            return True
        return False
class Engine:
    def __init__(self):
        self.a=0.6
        self.N=60
        self.sigma_target=200
        self.num_states=60
        self.vals=None
        self.vecs=None
        self.mask=None
        self.x=None
        self.y=None
        self.coeffs=None
        self.psi0=None
        self.time=0
        self.dt=0.01
        self.ipr=[]
        self.auto=[]
        self.is_solving=False
    def inside(self,X,Y):
        B=1
        in_rect=(np.abs(X)<=self.a)&(np.abs(Y)<=B)
        in_left=(X+self.a)**2+Y**2<=B**2
        in_right=(X-self.a)**2+Y**2<=B**2
        return in_rect|in_left|in_right
    def build(self):
        B=1
        nx=int(2*(self.a+B)*self.N)+1
        ny=int(2*B*self.N)+1
        x=np.linspace(-(self.a+B),(self.a+B),nx)
        y=np.linspace(-B,B,ny)
        dx=x[1]-x[0]
        dy=y[1]-y[0]
        X,Y=np.meshgrid(x,y)
        mask=self.inside(X,Y)
        idx=np.full(mask.shape,-1,int)
        idx[mask]=np.arange(mask.sum())
        rows,cols,data=[],[],[]
        rows_m,cols_m=np.where(mask)
        for r,c in zip(rows_m,cols_m):
            i=idx[r,c]
            rows.append(i); cols.append(i); data.append(-2/dx**2-2/dy**2)
            for dr,dc,d in [(0,1,dx),(0,-1,dx),(1,0,dy),(-1,0,dy)]:
                nr,nc=r+dr,c+dc
                if 0<=nr<ny and 0<=nc<nx and mask[nr,nc]:
                    j=idx[nr,nc]
                    rows.append(i); cols.append(j); data.append(1/d**2)
        L=csr_matrix((data,(rows,cols)),shape=(mask.sum(),mask.sum()))
        return L,mask,x,y
    def solve(self):
        self.is_solving=True
        try:
            L,mask,x,y=self.build()
            vals,vecs=eigsh(-L,k=self.num_states,sigma=self.sigma_target,which='LM')
            order=np.argsort(vals)
            self.vals=vals[order]
            self.vecs=vecs[:,order]
            self.mask=mask
            self.x=x
            self.y=y
        except:
            pass
        self.is_solving=False
    def start(self):
        if not self.is_solving:
            threading.Thread(target=self.solve,daemon=True).start()
    def initialize_packet(self,x0,y0,px,py,width=0.2):
        if self.vecs is None:
            return
        X,Y=np.meshgrid(self.x,self.y)
        psi0=np.exp(-((X-x0)**2+(Y-y0)**2)/(2*width**2))*np.exp(1j*(px*X+py*Y))
        psi0[~self.mask]=0
        dx=self.x[1]-self.x[0]
        dy=self.y[1]-self.y[0]
        self.coeffs=np.zeros(self.num_states,dtype=complex)
        for n in range(self.num_states):
            psi_n=np.zeros(self.mask.shape)
            psi_n[self.mask]=self.vecs[:,n]
            self.coeffs[n]=np.sum(np.conj(psi_n)*psi0)*dx*dy
        self.psi0=psi0
        self.time=0
        self.ipr.clear()
        self.auto.clear()
    def evolve(self):
        if self.coeffs is None:
            return np.zeros(self.mask.shape)
        psi=np.zeros(self.mask.shape,dtype=complex)
        for n in range(self.num_states):
            phase=np.exp(-1j*self.vals[n]*self.time)
            psi_n=np.zeros(self.mask.shape)
            psi_n[self.mask]=self.vecs[:,n]
            psi+=self.coeffs[n]*phase*psi_n
        dens=np.abs(psi)**2
        dx=self.x[1]-self.x[0]
        dy=self.y[1]-self.y[0]
        self.ipr.append(np.sum(dens**2)*dx*dy)
        self.auto.append(np.abs(np.sum(np.conj(self.psi0)*psi)*dx*dy)**2)
        self.time+=self.dt
        return dens
class QuantumLab:
    def __init__(self):
        pygame.init()
        self.screen=pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption("Quantum Scar Laboratory")
        self.clock=pygame.time.Clock()
        self.engine=Engine()
        self.engine.start()
        self.cmap=matplotlib.colormaps["inferno"]
        self.playing=False
        self.sliders=[
            Slider(20,100,250,0,1.5,self.engine.a,"Geometry a"),
            Slider(20,180,250,40,120,self.engine.N,"Resolution N"),
            Slider(20,260,250,50,500,self.engine.sigma_target,"Energy σ"),
            Slider(20,340,250,0.002,0.05,self.engine.dt,"Time step")
        ]
    def draw_sim(self,dens):
        if dens is None:
            return
        vmax=np.percentile(dens[self.engine.mask],98) if np.any(dens) else 1
        dens_norm=np.clip(dens/vmax,0,1)
        rgb=(self.cmap(dens_norm)[:,:,:3]*255).astype(np.uint8)
        surf=pygame.surfarray.make_surface(np.flipud(rgb))
        surf=pygame.transform.scale(surf,(SIM_WIDTH,SIM_HEIGHT))
        self.screen.blit(surf,(UI_WIDTH,0))
    def draw_metrics(self):
        panel=pygame.Rect(UI_WIDTH,SIM_HEIGHT,SIM_WIDTH,250)
        pygame.draw.rect(self.screen,WHITE,panel)
        pygame.draw.rect(self.screen,GRAY,panel,1)
        if len(self.engine.ipr)<2:
            return
        ipr=self.engine.ipr[-300:]
        auto=self.engine.auto[-300:]
        max_ipr=max(ipr) if max(ipr)!=0 else 1
        max_auto=max(auto) if max(auto)!=0 else 1
        w=panel.width-40
        h=panel.height-40
        def plot(data,max_val,color):
            pts=[]
            for i,val in enumerate(data):
                x=panel.x+20+i/len(data)*w
                y=panel.y+20+h-(val/max_val)*h
                pts.append((x,y))
            if len(pts)>1:
                pygame.draw.lines(self.screen,color,False,pts,2)
        plot(ipr,max_ipr,GREEN)
        plot(auto,max_auto,BLUE)
        self.screen.blit(FONT.render("Green: IPR",True,DARK),(panel.x+20,panel.y+5))
        self.screen.blit(FONT.render("Blue: Autocorrelation",True,DARK),(panel.x+150,panel.y+5))
    def run(self):
        dragging=False
        drag_start=None
        while True:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    pygame.quit(); sys.exit()
                for s in self.sliders:
                    if s.handle(event):
                        self.engine.a=self.sliders[0].value
                        self.engine.N=int(self.sliders[1].value)
                        self.engine.sigma_target=self.sliders[2].value
                        self.engine.dt=self.sliders[3].value
                if event.type==pygame.MOUSEBUTTONDOWN and event.pos[0]>UI_WIDTH:
                    dragging=True
                    drag_start=event.pos
                if event.type==pygame.MOUSEBUTTONUP and dragging:
                    dragging=False
                    if self.engine.vecs is not None:
                        end=event.pos
                        x0,y0=self.screen_to_world(drag_start)
                        px=(end[0]-drag_start[0])*0.02
                        py=(end[1]-drag_start[1])*0.02
                        self.engine.initialize_packet(x0,y0,px,-py)
                        self.playing=True
                if event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_SPACE:
                        self.playing=not self.playing
                    if event.key==pygame.K_r:
                        self.engine.start()
            self.screen.fill(WHITE)
            for s in self.sliders:
                s.draw(self.screen)
            self.screen.blit(FONT_LG.render("Controls",True,DARK),(20,40))
            dens=None
            if not self.engine.is_solving and self.engine.vecs is not None:
                dens=self.engine.evolve() if self.playing else np.zeros(self.engine.mask.shape)
            if dens is not None:
                self.draw_sim(dens)
                self.draw_metrics()
            pygame.display.flip()
    def screen_to_world(self,pos):
        xpix,ypix=pos
        B=1
        a=self.engine.a
        sx=SIM_WIDTH/(2*(a+B))
        sy=SIM_HEIGHT/(2*B)
        x=(xpix-UI_WIDTH)/sx-(a+B)
        y=(SIM_HEIGHT-ypix)/sy-B
        return x,y
if __name__=="__main__":
    QuantumLab().run()
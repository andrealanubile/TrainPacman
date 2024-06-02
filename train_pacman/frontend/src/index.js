import './styles.css';
import { Application, Graphics, Assets, Sprite } from 'pixi.js';
import MazeSprites from './mazesprites';
import Pellets from './pellets';
import PacmanSprites from './pacmansprites';
import GhostSprites from './ghostsprites';
import * as constants from './constants';
import LifeSprites from './lifesprites';

(async () =>
{
    const app = new Application();

    globalThis.__PIXI_APP__ = app;
    
    await app.init({
        canvas: document.getElementById('pixiCanvas'),
        width: constants.SCREENWIDTH,
        height: constants.SCREENHEIGHT,
        background: 'black',
    });

    const mazefile = await fetch('assets/maze1.txt').then(response => response.text());
    const rotfile = await fetch('assets/maze1_rotation.txt').then(response => response.text());
    const pelletfile = await fetch('assets/pellets_mask.txt').then(response => response.text());

    const maze = new MazeSprites(mazefile, rotfile);
    await maze.loadSpritesheet();
    const background = maze.constructBackground(0);
    app.stage.addChild(background);

    const pellets = new Pellets(pelletfile);
    const pelletsGraphics = pellets.drawPellets()
    app.stage.addChild(pelletsGraphics);

    const pacman = new PacmanSprites();
    await pacman.loadSpritesheet();
    const pacmanSprite = pacman.drawPacman(0);
    app.stage.addChild(pacmanSprite);

    const ghost = new GhostSprites('INKY', constants.CHASE, constants.RIGHT);
    await ghost.loadSpritesheet();
    const ghostSprite = ghost.drawGhost();
    app.stage.addChild(ghostSprite);

    const lives = new LifeSprites();
    await lives.loadSpritesheet();
    lives.resetLives(4);
    const livesContainer = lives.drawLives();
    app.stage.addChild(livesContainer);


})();


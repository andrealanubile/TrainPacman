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
    const apiBaseUrl = process.env.APP_API_URL || 'train-pacman.com'
    const app = new Application();

    globalThis.__PIXI_APP__ = app;
    
    await app.init({
        canvas: document.getElementById('pixiCanvas'),
        width: constants.SCREENWIDTH,
        height: constants.SCREENHEIGHT,
        background: 'black',
    });

    let data;
    try {
        const response = await fetch(`http://${apiBaseUrl}/api/initial-state/`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        data = await response.json();
    } catch (fetchError) {
        console.error('Error fetching initial coordinate:', fetchError);
    }
    let pacman_loc = JSON.parse(data['pacman_loc']);

    const mazefile = await fetch('assets/maze0.txt').then(response => response.text());
    const rotfile = await fetch('assets/maze0_rotation.txt').then(response => response.text());
    const pelletfile = await fetch('assets/pellets_mask.txt').then(response => response.text());

    const maze = new MazeSprites(mazefile, rotfile);
    await maze.loadSpritesheet();
    const background = maze.constructBackground(0);
    app.stage.addChild(background);

    let pellets = new Pellets(pelletfile);
    app.stage.addChild(pellets.pellets);

    const pacman = new PacmanSprites();
    await pacman.loadSpritesheet();
    const pacmanSprite = pacman.getSprite();
    app.stage.addChild(pacmanSprite);
    pacman.drawPacman(pacman_loc[0], pacman_loc[1], 0);
    // let sprite = pacman.getSprite();
    // app.stage.addChild(sprite);
    // pacman.drawPacman(0, 0, 0);
    // pacman.changeImage();

    // const ghost = new GhostSprites('INKY', constants.CHASE, constants.RIGHT);
    // await ghost.loadSpritesheet();
    // const ghostSprite = ghost.drawGhost();
    // app.stage.addChild(ghostSprite);

    const lives = new LifeSprites();
    await lives.loadSpritesheet();
    lives.resetLives(4);
    const livesContainer = lives.drawLives();
    app.stage.addChild(livesContainer);

    // Connect to WebSocket
    const socket = new WebSocket(`ws://${apiBaseUrl}/ws/pacman/`);

    socket.onopen = () => {
        console.log('WebSocket connection established');
    };

    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    socket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            pacman_loc = JSON.parse(data['pacman_loc']);
            pacman.direction = data['pacman_direction'];
            const pelletList = JSON.parse(data['pellets']);
            // pellets.pelletList = pelletList;
            pellets.setPelletList(pelletList);

            pellets.drawPellets();
            pacman.drawPacman(pacman_loc[0], pacman_loc[1], 1);
        } catch (parseError) {
            console.error('Error parsing WebSocket message:', parseError)
        }
    };


})();


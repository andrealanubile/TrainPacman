import './styles.css';
import { Application, Graphics, Assets, Sprite } from 'pixi.js';
import MazeSprites from './mazesprites';
import Pellets from './pellets';
import PacmanSprites from './pacmansprites';
import GhostSprites from './ghostsprites';
import * as constants from './constants';
import LifeSprites from './lifesprites';

function resizeCanvas(app) {
    const container = document.getElementById('pixi-container');
    const canvas = document.getElementById('pixiCanvas');
    const controls = document.getElementById('controls');

    const containerWidth = container.clientWidth;
    // const containerHeight = window.innerHeight - controls.offsetHeight - 20; // Adjust for controls and margin
    const containerHeight = container.clientHeight;

    canvas.style.width = `${containerWidth}px`;
    canvas.style.height = `${containerHeight}px`;
    app.renderer.resize(containerWidth, containerHeight);
}

(async () =>
{
    const apiBaseUrl = process.env.APP_API_URL || 'train-pacman.com'
    const app = new Application();

    globalThis.__PIXI_APP__ = app;

    const pixiContainer = document.getElementById('pixi-container');
    const pixiCanvas = document.getElementById('pixiCanvas');
    const { width, height } = pixiContainer.getBoundingClientRect();

    const scaleX = width / constants.SCREENWIDTH;
    const scaleY = height / constants.SCREENHEIGHT;

    await app.init({
        canvas: pixiCanvas,
        width: width,
        height: height,
        // width: constants.SCREENWIDTH,
        // height: constants.SCREENHEIGHT,
        resizeTo: pixiContainer,
        autoDensity: true,
        background: 'black',
    });

    // window.addEventListener('resize', () => resizeCanvas(app));
    // resizeCanvas(app);

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
    let ghost_loc = JSON.parse(data['ghost_loc']);

    const mazefile = await fetch('assets/maze1.txt').then(response => response.text());
    const rotfile = await fetch('assets/maze1_rotation.txt').then(response => response.text());

    const maze = new MazeSprites(mazefile, rotfile, scaleX, scaleY);
    await maze.loadSpritesheet();
    maze.constructBackground(0);
    let background = maze.getBackground();
    app.stage.addChild(background);

    let pellets = new Pellets();
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

    const ghost = new GhostSprites('BLINKY', constants.CHASE, constants.RIGHT);
    await ghost.loadSpritesheet();
    const ghostSprite = ghost.getSprite();
    app.stage.addChild(ghostSprite);
    ghost.drawGhost(ghost_loc[0], ghost_loc[1], 0);

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
            ghost_loc = JSON.parse(data['ghost_loc']);
            ghost.direction = data['ghost_direction'];
            const pelletList = JSON.parse(data['pellets']);
            // pellets.pelletList = pelletList;
            pellets.setPelletList(pelletList);

            pellets.drawPellets();
            pacman.drawPacman(pacman_loc[0], pacman_loc[1], 1);
            ghost.drawGhost(ghost_loc[0], ghost_loc[1], 1);
        } catch (parseError) {
            console.error('Error parsing WebSocket message:', parseError)
        }
    };

    document.getElementById('button_plus1').addEventListener('click', () => {
        socket.send(JSON.stringify({ action: 'reward_plus1' }));
    });
    document.getElementById('button_plus10').addEventListener('click', () => {
        socket.send(JSON.stringify({ action: 'reward_plus10' }));
    });
    document.getElementById('button_minus1').addEventListener('click', () => {
        socket.send(JSON.stringify({ action: 'reward_minus1' }));
    });
    document.getElementById('button_minus10').addEventListener('click', () => {
        socket.send(JSON.stringify({ action: 'reward_minus10' }));
    });


    const resize = () => {
        const { width, height } = pixiContainer.getBoundingClientRect();
        app.renderer.resize(width, height);

        const scaleX = width / constants.SCREENWIDTH;
        const scaleY = height / constants.SCREENHEIGHT;

        console.log(width)
        console.log(height)
        maze.updateScale(scaleX, scaleY);
        maze.updateBackground();
    };

    window.addEventListener('resize', resize);



})();


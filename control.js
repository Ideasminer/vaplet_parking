var goback = document.querySelector("#goback");
var side = document.querySelector(".side");
var func = document.querySelector(".func");
var show = document.querySelector(".show");
var gopre = document.querySelector("#gopre");
var renderbtn = document.querySelector(".renderbox");
var layout = document.querySelector("#layout");
var download = document.querySelector(".download");
var modalclose = document.querySelectorAll("#close");
var loginbox = document.querySelector("#loginbox");
var historybox = document.querySelector("#historybox");
var avatorbox = document.querySelector("#avatorbox");
var configbox = document.querySelector("#configbox");
var refreshbox = document.querySelector("#refreshbox");
var toregister = document.querySelector(".inform");
var register = document.querySelector("#register");
var allModal = document.querySelectorAll(".modal");
var mask = document.querySelector(".mask");
var github = document.querySelector(".description .icon");
var hidden_submit = document.querySelector("#hidden_submit");
var sendInfo = document.querySelector(".sendlogin");
var modals = new Array();

allModal.forEach(function(e){
    modals.push(e);
})
goback.addEventListener("click", function (e) {
    side.style.transition = "0.3s";
    func.style.transition = "0.3s";
    show.style.transition = "0.3s";
    side.style.left = "-8vw";
    func.style.left = "0vw";
    show.style.left = "0vw";
    func.style.width = "98vw";
    show.style.width = "98vw";
    goback.style.display = "none";
    gopre.style.display = "inline";
})
gopre.addEventListener("click", function (e) {
    side.style.transition = "0.3s";
    func.style.transition = "0.3s";
    show.style.transition = "0.3s";
    side.style.left = "0vw";
    func.style.left = "8vw";
    show.style.left = "8vw";
    func.style.width = "90vw";
    show.style.width = "90vw";
    gopre.style.display = "none";
    goback.style.display = "inline";
})
renderbtn.addEventListener("click", () => {
    renderbtn.style.display = "none";
    download.style.display = "block";
    layout.style.display = "inline";

});
modalclose.forEach(function(e){
    e.addEventListener("click", () => {
        mask.style.display="none";
        e.parentElement.style.display = "none";
    })
});
function switchModal(e, modal = modals){
    var position = modal.indexOf(e);
    e.style.display = "block";
    mask.style.display="block"
    modal.splice(position, 1);
    modal.forEach(function(n){
        if(n.style.display === "block"){
            n.style.display="none";
        }
    });
    modal.push(e);
}
loginbox.addEventListener("click", function(e) {
    switchModal(document.querySelector("#login"));
});
historybox.addEventListener("click", function(e) {
    switchModal(document.querySelector("#history"));
});
configbox.addEventListener("click", function(e) {
    switchModal(document.querySelector("#setting"));
});
avatorbox.addEventListener("click", function(e) {
    switchModal(document.querySelector("#usr"));
});
refreshbox.addEventListener("click", function(e) {
    window.open("http://127.0.0.1:8000", '_self')
});
toregister.addEventListener("click", function(e) {
    toregister.parentElement.parentElement.style.display="none";
    register.style.display="block";
});
github.addEventListener("click", function(e) {
    window.open("https://github.com/Ideasminer/VAPSIM")
});

sendInfo.addEventListener("click", function(e) {
    hidden_submit.click();
});

